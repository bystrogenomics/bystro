use 5.10.0;
use strict;
use warnings;

package Seq::DBManager;

our $VERSION = '0.001';

# ABSTRACT: Manages Database connection
# VERSION

#TODO: Better errors; Seem to get bad perf if copy error after each db call
#TODO: Allow missing database only if in $dbReadOnly mode
#TODO: Better singleton handling

use Mouse 2;
with 'Seq::Role::Message';

use Data::MessagePack;
use LMDB_File qw(:all);
use Types::Path::Tiny qw/AbsPath/;
use DDP;
use Hash::Merge::Simple qw/ merge /;
use Path::Tiny;
use Scalar::Util qw/looks_like_number/;

# We will maintain our own, internal database log for errors
use Cwd;
use Log::Fast;

# Most common error is "MDB_NOTFOUND" which isn't nec. bad.
$LMDB_File::die_on_err = 0;

######### Public Attributes
# Flag for deleting tracks instead of inserting during patch* methods
has delete => (is => 'rw', isa => 'Bool', default => 0, lazy => 1);

has dryRun => (is => 'rw', isa => 'Bool', default => 0, lazy => 1);

# DBManager maintains its own, internal log, so that in a multi-user environment
# DBA can keep track of key errors
# TODO: Warning: cwd may fill up if left unchecked
my $internalLog = Log::Fast->new({
  path            => path( getcwd() )->child('dbManager-error.log')->stringify,
  pid             => $$,
});

# instanceConfig variable holding our databases; this way can be used
# in environment where calling process never dies
# {
#   database_dir => {
#     env => $somEnv, dbi => $someDbi
#   }
# }
####################### Static Properties ############################
my $instance;
# instanceConfig contains
# databaseDir => <Path::Tiny>
# readOnly => <Bool>
my %instanceConfig;
# Each process should open each environment only once.
# http://www.lmdb.tech/doc/starting.html
my %envs;
# We are enforcing a singel transaction per environment for the moment, especially
# in light of the apparent LMDB_File restriction to this effect
my %cursors;

# Can call as class method (DBManager->setDefaultDatabaseDir), or as instanceConfig method
# Prepares the class for consumption; should be run before the program can fork
# To ensure that all old data is cleared, if executing from a long-running process
sub initialize {
  my $data = @_ == 2 ? $_[1] : $_[0];

  if(%instanceConfig) {
     $internalLog->WARN("dbManager already initialized; clearing state");
  }

  cleanUp();

  undef $instance;
  undef %instanceConfig;
  undef %envs;
  undef %cursors;

  if(!$data->{databaseDir}) {
     $internalLog->ERR("dbManager requires a databaseDir");
     die;
  }

  $instanceConfig{databaseDir} = path($data->{databaseDir});
  if(!$instanceConfig{databaseDir}->exists) { $instanceConfig{databaseDir}->mkpath; }

  if($data->{readOnly}) {
    $instanceConfig{readOnly} = 1;
  }

  shift;
  return __PACKAGE__->new(@_);
}

around 'new' => sub {
  my $orig = shift;
  my $self = shift;

  return $instance //= $self->$orig(@_)
};

sub BUILD {
  my $self = shift;

  # TODO: think about better way to initialize this class w.r.t databaseDir
  if(!$instanceConfig{databaseDir}) {
    $self->_errorWithCleanup("DBManager requires databaseDir");
  }

  if(!$instanceConfig{databaseDir}->is_dir) { $self->_errorWithCleanup('databaseDir not a directory'); }
};

# Our packing function
#treat "1" as an integer, save more space
#treat .00012 as a single precision float, saving 4 bytes.
my $mp = Data::MessagePack->new()->prefer_integer()->prefer_float32();

################### DB Read, Write methods ############################
# Unsafe for $_[2] ; will be modified if an array is passed
# Read transactions are committed by default
sub dbReadOne {
  #my ($self, $chr, $posAref, $skipCommit, $stringKeys,) = @_;
  #== $_[0], $_[1], $_[2],    $_[3],       $_[4] (don't assign to avoid copy)

  #It is possible not to find a database in $dbReadOnly mode (for ex: refSeq for a while didn't have chrM)
  #http://ideone.com/uzpdZ8
  #                      #$name, $dontCreate, $stringKeys
  my $db = $_[0]->_getDbi($_[1], 1, $_[4]) or return;

  if(!$db->{db}->Alive) {
    $db->{db}->Txn = $db->{env}->BeginTxn();
    # not strictly necessary, but I am concerned about hard to trace abort bugs related to scope
    $db->{db}->Txn->AutoCommit(1);
  }

  $db->{db}->Txn->get($db->{dbi}, $_[2], my $json);

  # Commit unless the user specifically asks not to
  #if(!$skipCommit) {
  $db->{db}->Txn->commit() unless $_[3];

  if($LMDB_File::last_err) {
    if($LMDB_File::last_err != MDB_NOTFOUND ) {
      $_[0]->_errorWithCleanup("dbRead LMDB error $LMDB_File::last_err");
      return 255;
    }

    $LMDB_File::last_err = 0;
  }

  return defined $json ? $mp->unpack($json) : undef;
}

# Unsafe for $_[2] ; will be modified if an array is passed
# Read transactions are committed by default
sub dbRead {
  #my ($self, $chr, $posAref, $skipCommit, $stringKeys) = @_;
  #== $_[0], $_[1], $_[2],    $_[3],       $_[4] (don't assign to avoid copy)
  if(!ref $_[2]) {
    goto &dbReadOne;
  }

  #It is possible not to find a database in dbReadOnly mode (for ex: refSeq for a while didn't have chrM)
  #http://ideone.com/uzpdZ8
  #                      #$name, $dontCreate, $stringKeys
  my $db = $_[0]->_getDbi($_[1], 0, $_[4]) or return [];
  my $dbi = $db->{dbi};

  if(!$db->{db}->Alive) {
    $db->{db}->Txn = $db->{env}->BeginTxn();
    # not strictly necessary, but I am concerned about hard to trace abort bugs related to scope
    $db->{db}->Txn->AutoCommit(1);
  }

  my $txn = $db->{db}->Txn;

  my $json;

  # Modifies $posAref ($_[2]) to avoid extra allocation
  for my $pos (@{ $_[2] }) {
    $txn->get($dbi, $pos, $json);

    $pos = defined $json ? $mp->unpack($json) : undef;
  }

  # Commit unless the user specifically asks not to
  #if(!$skipCommit) {
  $txn->commit() unless $_[3];

  #substantial to catch any errors
  if($LMDB_File::last_err) {
    if($LMDB_File::last_err != MDB_NOTFOUND) {
      $_[0]->_errorWithCleanup("dbRead LMDB error after loop: $LMDB_File::last_err");
      return 255;
    }

    #reset the class error variable, to avoid crazy error reporting later
    $LMDB_File::last_err = 0;
  }

  #will return a single value if we were passed one value
  #return \@out;
  return $_[2];
}

#Assumes that the posHref is
# {
#   position => {
#     feature_name => {
#       ...everything that belongs to feature_name
#     }
#   }
# }

# Method to write one key => value pair to the database, as a hash
# $pos can be any string, identifies a key within the kv database
# dataHref should be {someTrackName => someData} that belongs at $chr:$pos
# Currently only used for region tracks (currently only the Gene Track region track)
sub dbPatchHash {
  my ($self, $chr, $pos, $dataHref, $mergeFunc, $skipCommit, $overwrite, $stringKeys) = @_;

  if(ref $dataHref ne 'HASH') {
    $self->_errorWithCleanup("dbPatchHash requires a 1-element hash of a hash");
    return 255;
  }

  # 0 argument means "create if not found"
  # last argument means we want string keys rather than integer keys
  my $db = $self->_getDbi($chr, 0, $stringKeys);

  if(!$db) {
    $self->_errorWithCleanup("Couldn't open $chr database. readOnly is " . ($instanceConfig{readOnly} ? "set" : "not set"));
    return 255;
  }

  my $dbi = $db->{dbi};

  if(!$db->{db}->Alive) {
    $db->{db}->Txn = $db->{env}->BeginTxn();
    # not strictly necessary, but I am concerned about hard to trace abort bugs related to scope
    $db->{db}->Txn->AutoCommit(1);
  }

  #zero-copy read, don't modify $json
  $db->{db}->Txn->get($dbi, $pos, my $json);

  if($LMDB_File::last_err && $LMDB_File::last_err != MDB_NOTFOUND) {
    $self->_errorWithCleanup("dbPatchHash LMDB error during get: $LMDB_File::last_err");
    return 255;
  }

  $LMDB_File::last_err = 0;

  my $href;
  my $skip;
  if($json) {
    $href = $mp->unpack($json);

    my ($trackKey, $trackValue) = %{$dataHref};

    if(!defined $trackKey || ref $trackKey ) {
      $self->_errorWithCleanup("dbPatchHash requires scalar trackKey");
      return 255;
    }

    # Allows undefined trackValue

    if(defined $href->{$trackKey}) {
      # Deletion and insertion are mutually exclusive
      if($self->delete) {
        delete $href->{$trackKey};
      } elsif($overwrite) {
        # Merge with righthand hash taking precedence, https://ideone.com/SBbfYV
        # Will overwrite any keys of the same name (why it's called $overwrite)
        $href = merge $href, $dataHref;
      } elsif(defined $mergeFunc) {
        (my $err, $href->{$trackKey}) = &$mergeFunc($chr, $pos, $href->{$trackKey}, $trackValue);

        if($err) {
          $self->_errorWithCleanup("dbPatchHash mergeFunc error: $err");
          return 255;
        }
      } else {
        # Nothing to do, value exists, we're not deleting, overwrite, or merging
        $skip = 1;
      }
    } elsif($self->delete) {
      # We want to delete a non-existant key, skip
      $skip = 1;
    } else {
      $href->{$trackKey} = $trackValue;
    }
  } elsif($self->delete) {
    # If we want to delete, and no data, there's nothing to do, skip
    $skip = 1;
  }

  #insert href if we have that (only truthy if defined), or the data provided as arg
  if(!$skip) {
    if($self->dryRun) {
      $self->log('info', "DBManager dry run: would have dbPatchHash $chr\:$pos");
    } else {
      $db->{db}->Txn->put($db->{dbi}, $pos, $mp->pack($href || $dataHref));
    }
  }

  $db->{db}->Txn->commit() unless $skipCommit;

  if($LMDB_File::last_err) {
    if($LMDB_File::last_err != MDB_KEYEXIST) {
      $self->_errorWithCleanup("dbPut LMDB error: $LMDB_File::last_err");
      return 255;
    }

    #reset the class error variable, to avoid crazy error reporting later
    $LMDB_File::last_err = 0;
  }


  return 0;
}

#Method to write a single position into the main databse
# Write transactions are by default committed
# Removed delete, overwrite capacities
sub dbPatch {
  #my ($self, $chr, $trackIndex, $pos, $trackValue, $mergeFunc, $skipCommit, $stringKeys) = @_;
  #.   $_[0], $_[1] $_[2]        $_[3] $_[4]        $_[5]       $_[6]        $_[7]

  # 0 argument means "create if not found"
  #my $db = $self->_getDbi($chr, 0, $stringKeys);
  my $db = $_[0]->_getDbi($_[1], 0, $_[7]) or return 255;

  if(!$db) {
    $_[0]->_errorWithCleanup("Couldn't open $_[1] database. readOnly is " . ($instanceConfig{readOnly} ? "set" : "not set"));
    return 255;
  }

  if(!$db->{db}->Alive) {
    $db->{db}->Txn = $db->{env}->BeginTxn();
    # not strictly necessary, but I am concerned about hard to trace abort bugs related to scope
    $db->{db}->Txn->AutoCommit(1);
  }

  my $txn = $db->{db}->Txn;

  #zero-copy
 #$db->{db}->Txn->get($db->{dbi}, $pos, my $json);
  $txn->get($db->{dbi}, $_[3], my $json);

  if($LMDB_File::last_err) {
    if($LMDB_File::last_err != MDB_NOTFOUND) {
     #$self->
      $_[0]->_errorWithCleanup("dbPatch LMDB error $LMDB_File::last_err");
      return 255;
    }

    #reset the class error variable, to avoid crazy error reporting later
    $LMDB_File::last_err = 0;
  }

  my $aref = defined $json ? $mp->unpack($json) : [];

  #Undefined track values are allowed as universal-type "missing data" signal
            #$aref->[$trackIndex]
  if(defined $aref->[$_[2]]) {
    #if($mergeFunc) {
    if($_[5]) {
        #$aref->[$trackIndex]) = $mergeFunc->($chr, $pos, $aref->[$trackIndex], $trackValue);
      (my $err, $aref->[$_[2]]) = $_[5]->($_[1], $_[3], $aref->[$_[2]], $_[4]);

      if($err) {
        #$self
        $_[0]->_errorWithCleanup("mergeFunc error: $err");
        return 255;
      }

      # Nothing to update
      if(!defined $aref->[$_[2]]) {
        return 0;
      }
    } else {
      # No overriding
      return 0;
    }
  } else {
          #[$trackIndex] = $trackValue
    $aref->[$_[2]] = $_[4];
  }

 #if($self->dryRun) {
  if($_[0]->dryRun) {
  #$self->
    $_[0]->log('info', "DBManager dry run: would have dbPatch $_[1]\:$_[3]");
  } else {
   #$txn->put($db->{dbi}, $pos, $mp->pack($aref));
    $txn->put($db->{dbi}, $_[3], $mp->pack($aref));
  }

 #if(!$skipCommit) {
  $txn->commit() unless $_[6];

  if($LMDB_File::last_err) {
    if($LMDB_File::last_err != MDB_KEYEXIST) {
     #$self->
      $_[0]->_errorWithCleanup("dbPatch put or commit LMDB error $LMDB_File::last_err");
      return 255;
    }

    #reset the class error variable, to avoid crazy error reporting later
    $LMDB_File::last_err = 0;
  }

  return 0;
}

# Write transactions are by default committed
sub dbPut {
  my ($self, $chr, $pos, $data, $skipCommit, $stringKeys) = @_;

  if($self->dryRun) {
    $self->log('info', "DBManager dry run: would have dbPut $chr:$pos");
    return 0;
  }

  if(!(defined $chr && defined $pos)) {
    $self->_errorWithCleanup("dbPut requires position");
    return 255;
  }

  # 0 to create database if not found
  my $db = $self->_getDbi($chr, 0, $stringKeys);

  if(!$db) {
    $self->_errorWithCleanup("Couldn't open $chr database. readOnly is " . ($instanceConfig{readOnly} ? "set" : "not set"));
    return 255;
  }

  if(!$db->{db}->Alive) {
    $db->{db}->Txn = $db->{env}->BeginTxn();
    # not strictly necessary, but I am concerned about hard to trace abort bugs related to scope
    $db->{db}->Txn->AutoCommit(1);
  }

  $db->{db}->Txn->put($db->{dbi}, $pos, $mp->pack( $data ) );

  $db->{db}->Txn->commit() unless $skipCommit;

  if($LMDB_File::last_err) {
    if($LMDB_File::last_err != MDB_KEYEXIST) {
      $self->_errorWithCleanup("dbPut LMDB error: $LMDB_File::last_err");
      return 255;
    }

    #reset the class error variable, to avoid crazy error reporting later
    $LMDB_File::last_err = 0;
  }

  return 0;
}

sub dbDelete {
  my ($self, $chr, $pos, $stringKeys) = @_;

  if($self->dryRun) {
    $self->log('info', "DBManager dry run: Would have dbDelete $chr\:$pos");
    return 0;
  }

  if(!(defined $chr && defined $pos)) {
    $self->_errorWithCleanup("dbDelete requires chr and position");
    return 255;
  }

  my $db = $self->_getDbi($chr, $stringKeys);

  if(!$db->{db}->Alive) {
    $db->{db}->Txn = $db->{env}->BeginTxn();
    # not strictly necessary, but I am concerned about hard to trace abort bugs related to scope
    $db->{db}->Txn->AutoCommit(1);
  }

  # Error with LMDB_File api, means $data is required as 3rd argument,
  # even if it is undef
  $db->{db}->Txn->del($db->{dbi}, $pos, undef);

  if($LMDB_File::last_err && $LMDB_File::last_err != MDB_NOTFOUND) {
    $self->_errorWithCleanup("dbDelete LMDB error: $LMDB_File::last_err");
    return 255;
  }

  $LMDB_File::last_err = 0;

  $db->{db}->Txn->commit();

  if($LMDB_File::last_err) {
    $self->_errorWithCleanup("dbDelete commit LMDB error: $LMDB_File::last_err");
    return 255;
  }

  #reset the class error variable, to avoid crazy error reporting later
  $LMDB_File::last_err = 0;
  return 0;
}

#cursor version
# Read transactions are by default not committed
sub dbReadAll {
  my ( $self, $chr, $skipCommit, $stringKeys) = @_;
  #==   $_[0], $_[1], $_[2]

  #It is possible not to find a database in dbReadOnly mode (for ex: refSeq for a while didn't have chrM)
  #http://ideone.com/uzpdZ8
  my $db = $self->_getDbi($chr, 0, $stringKeys) or return;

  if(!$db->{db}->Alive) {
    $db->{db}->Txn = $db->{env}->BeginTxn();
    # not strictly necessary, but I am concerned about hard to trace abort bugs related to scope
    $db->{db}->Txn->AutoCommit(1);
  }

  # We store data in sequential, integer order
  # in all but the meta tables, which don't use this function
  # LMDB::Cursor::open($txn, $db->{dbi}, my $cursor);
  my $cursor = $db->{db}->Cursor;

  my ($key, $value, @out);
  my $first = 1;
  while(1) {
    if($first) {
      $cursor->_get($key, $value, MDB_FIRST);
      $first = 0;
    } else {
      $cursor->_get($key, $value, MDB_NEXT);
    }

    #because this error is generated right after the get
    #we want to capture it before the next iteration
    #hence this is not inside while( )
    if($LMDB_File::last_err == MDB_NOTFOUND) {
      $LMDB_FILE::last_err = 0;
      last;
    }

    if($LMDB_FILE::last_err) {
      $_[0]->_errorWithCleanup("dbReadAll LMDB error $LMDB_FILE::last_err");
      return 255;
    }

    push @out, $mp->unpack($value);
  }

  #  !$skipCommit
  $db->{db}->Txn->commit() unless $skipCommit;

  if($LMDB_File::last_err) {
    if($LMDB_File::last_err != MDB_NOTFOUND) {
      $_[0]->_errorWithCleanup("dbReadAll LMDB error at end: $LMDB_File::last_err");
      return 255;
    }

    #reset the class error variable, to avoid crazy error reporting later
    $LMDB_File::last_err = 0;
  }

  return \@out;
}

# Delete all values within a database; a necessity if we want to update a single track
# TODO: this may inflate database size, because very long-lived transaction
# maybe should allow to commit
sub dbDeleteAll {
  my ( $self, $chr, $dbName, $stringKeys) = @_;

  #It is possible not to find a database in dbReadOnly mode (for ex: refSeq for a while didn't have chrM)
  #http://ideone.com/uzpdZ8
  my $db = $self->_getDbi($chr, 0, $stringKeys) or return;

  if(!$db->{db}->Alive) {
    $db->{db}->Txn = $db->{env}->BeginTxn();
    # not strictly necessary, but I am concerned about hard to trace abort bugs related to scope
    $db->{db}->Txn->AutoCommit(1);
  }

  # We store data in sequential, integer order
  # in all but the meta tables, which don't use this function
  # LMDB::Cursor::open($txn, $db->{dbi}, my $cursor);
  my $cursor = $db->{db}->Cursor;

  my ($key, $value, @out);
  my $first = 1;
  while(1) {
    if($first) {
      $cursor->_get($key, $value, MDB_FIRST);
      $first = 0;
    } else {
      $cursor->_get($key, $value, MDB_NEXT);
    }

    #because this error is generated right after the get
    #we want to capture it before the next iteration
    #hence this is not inside while( )
    if($LMDB_File::last_err == MDB_NOTFOUND) {
      $LMDB_FILE::last_err = 0;
      last;
    }

    if($LMDB_FILE::last_err) {
      $_[0]->_errorWithCleanup("dbReadAll LMDB error $LMDB_FILE::last_err");
      return 255;
    }

    my $vals = $mp->unpack($value);

    if($vals->[$dbName]) {
      $vals->[$dbName] = undef;

      $cursor->_put($key, $mp->pack($vals), MDB_CURRENT);
    }
  }

  #  !$skipCommit
  $db->{db}->Txn->commit();

  if($LMDB_File::last_err) {
    if($LMDB_File::last_err != MDB_NOTFOUND) {
      $_[0]->_errorWithCleanup("dbReadAll LMDB error at end: $LMDB_File::last_err");
      return 255;
    }

    #reset the class error variable, to avoid crazy error reporting later
    $LMDB_File::last_err = 0;
  }

  return 0;
}

################################################################################
###### For performance reasons we may want to manage our own transactions ######
######################## WARNING: *UNSAFE* #####################################
sub dbStartCursorTxn {
  my ( $self, $chr) = @_;

  if($cursors{$chr}) {
    return $cursors{$chr};
  }

  #It is possible not to find a database in $dbReadOnly mode (for ex: refSeq for a while didn't have chrM)
  #http://ideone.com/uzpdZ8

  #        $self->_getDbi($chr)
  my $db = $self->_getDbi($chr);

  # TODO: Better error handling; since a cursor may be used to read or write
  # in most cases a database not existing indicates we set readOnly or are  need to return an error if the database doesn't exist
  if(!$db) {
    $self->_errorWithCleanup("Couldn't open $chr database because it doesn't exist. readOnly is " . ($instanceConfig{readOnly} ? "set" : "not set"));
    return 255;
  }

  # TODO: Investigate why a subtransaction isn't successfully made
  # when using BeginTxn()
  # If we create a txn and assign it to DB->Txn, from $db->{env}, before creating a txn here
  # upon trying to use the parent transaction, we will get a crash (-30782 / BAD_TXN)
  # no such issue arises the other way around; i.e creating this transaction, then having
  # a normal DB->Txn created as a nested transaction
  if($db->{db}->Alive) {
    $self->_errorWithCleanup("DB alive when calling dbStartCursorTxn, LMDB_File allows only 1 txn per environment: commit DB->Txn before dbStartCursorTxn");
    return 255;
  }

  # Will throw errors saying "should be nested transaction" unlike env->BeginTxn();
  # to protect against the above BAD_TXN issue
  my $txn = LMDB::Txn->new($db->{env}, $db->{tflags});

  # my $txn = LMDB::Txn->new($db->{env});
  $txn->AutoCommit(1);

  # This means LMDB_File will not track our cursor, must close/delete manually
  LMDB::Cursor::open($txn, $db->{dbi}, my $cursor);

  # TODO: better error handling
  if(!$cursor) {
    $self->_errorWithCleanup("Couldn't open cursor for $_[1]");
    return 255;
  }

  # Unsafe, private LMDB_File method access but Cursor::open does not track cursors
  $LMDB::Txn::Txns{$$txn}{Cursors}{$$cursor} = 1;

  $cursors{$chr} = [$txn, $cursor];

  # We store data in sequential, integer order
  # in all but the meta tables, which don't use this function
  # LMDB::Cursor::open($txn, $db->{dbi}, my $cursor);
  return $cursors{$chr};
}

# Assumes user manages their own transactions
# Don't copy variables on the stack, since this may be called billions of times
sub dbReadOneCursorUnsafe {
  #my ($self, $cursor, $pos) = @_;
      #$_[0]. $_[1].   $_[2]

  #$cursor->[1]->_get($pos)
  $_[1]->[1]->_get($_[2], my $json, MDB_SET);

  if($LMDB_File::last_err) {
    if($LMDB_File::last_err != MDB_NOTFOUND) {
     #$self->_errorWithCleanup
      $_[0]->_errorWithCleanup("dbEndCursorTxn LMDB error: $LMDB_File::last_err");
      return 255;
    }

    #reset the class error variable, to avoid crazy error reporting later
    $LMDB_File::last_err = 0;
  }

  return defined $json ? $mp->unpack($json) : undef;
}

# Don't copy variables on the stack, since this may be called billions of times
sub dbReadCursorUnsafe {
  #my ($self, $cursor, $posAref) = @_;
      #$_[0]. $_[1].   $_[2]

  foreach (@{$_[2]}) {
    $_[1]->[1]->_get($_, my $json, MDB_SET);

    $_ = defined $json ? $mp->unpack($json) : undef;
  }

  if($LMDB_File::last_err) {
    if($LMDB_File::last_err != MDB_NOTFOUND) {
    #$self
      $_[0]->_errorWithCleanup("dbEndCursorTxn LMDB error: $LMDB_File::last_err");
      return 255;
    }

    #reset the class error variable, to avoid crazy error reporting later
    $LMDB_File::last_err = 0;
  }

  return $_[2];
}

# When you need performance, especially for genome-wide insertions
# Be an adult, manage your own cursor
# LMDB tells you : if you commit the cursor is closed, needs to be renewed
# Don't copy variables on the stack, since this may be called billions of times
sub dbPatchCursorUnsafe {
  #my ( $self, $cursor, $chr, $dbName, $pos, $newValue, $mergeFunc) = @_;
  #    $_[0]. $_[1].   $_[2]. $_[3].  $_[4] $_[5].     $_[6]

#$cursor->_get($pos)
  $_[1]->[1]->_get($_[4], my $json, MDB_SET);

  my $existingValue = defined $json ? $mp->unpack($json) : [];
#                            [$dbName]
  if(defined $existingValue->[$_[3]]) {
    # ($mergeFunc)
    if($_[6]) {                #[$dbName]=$mergeFunc->($chr, $pos, $existingValue->[$dbName], $newValue);
      (my $err, $existingValue->[$_[3]]) = $_[6]->($_[2], $_[4], $existingValue->[$_[3]], $_[5]);

      if($err) {
        $_[0]->_errorWithCleanup("dbPatchCursor mergeFunc error: $err");
        return 255;
      }

      # nothing to do; no value returned
      if(!defined $existingValue->[$_[3]]) {
        return 0;
      }
    } else {
      # No overwrite allowed by default
      # just like dbPatch, but no overwrite option
      # Overwrite is impossible when mergeFunc is defined
      # TODO: remove overwrite from dbPatch
      return 0;
    }
  } else {
                  ##[$dbName]#$newValue
    $existingValue->[$_[3]] = $_[5];
  }

  #_put as used here will not return errors if the cursor is inactive
  # hence, "unsafe"
  if(defined $json) {
  #$cursor      #$pos
    $_[1]->[1]->_put($_[4], $mp->pack($existingValue), MDB_CURRENT);
  } else {
  #$cursor     #$pos
    $_[1]->[1]->_put($_[4], $mp->pack($existingValue));
  }

  if($LMDB_File::last_err) {
    if($LMDB_File::last_err != MDB_NOTFOUND && $LMDB_File::last_err != MDB_KEYEXIST) {
    #$self
      $_[0]->_errorWithCleanup("dbEndCursorTxn LMDB error: $LMDB_File::last_err");
      return 255;
    }

    #reset the class error variable, to avoid crazy error reporting later
    $LMDB_File::last_err = 0;
  }

  return 0;
}

# commit and close a self-managed cursor object
# TODO: Don't close cursor if not needed
sub dbEndCursorTxn {
  my ( $self, $chr ) = @_;

  if(!defined $cursors{$chr}) {
    return 0;
  }

  $cursors{$chr}->[1]->close();

  # closes a write cursor as well; the above $cursor->close() is to be explicit
  # will not close a MDB_RDONLY cursor
  $cursors{$chr}->[0]->commit();

  delete $cursors{$chr};

  # Allow two relatively innocuous errors, kill for anything else
  if($LMDB_File::last_err) {
    if($LMDB_File::last_err != MDB_NOTFOUND && $LMDB_File::last_err != MDB_KEYEXIST) {
      $self->_errorWithCleanup("dbEndCursorTxn LMDB error: $LMDB_File::last_err");
      return 255;
    }

    #reset the class error variable, to avoid crazy error reporting later
    $LMDB_File::last_err = 0;
  }

  return 0;
}

################################################################################

sub dbGetNumberOfEntries {
  my ( $self, $chr ) = @_;

  #get database, but don't create it if it doesn't exist
  my $db = $self->_getDbi($chr,1);

  return $db ? $db->{env}->stat->{entries} : 0;
}

#to store any records
#For instanceConfig, here we can store our feature name mappings, our type mappings
#whether or not a particular track has completed writing, etc
state $metaDbNamePart = '_meta';

#We allow people to update special "Meta" databases
#The difference here is that for each $databaseName, there is always
#only one meta database. Makes storing multiple meta documents in a single
#meta collection easy
#For example, users may want to store field name mappings, how many rows inserted
#whether building the database was a success, and more
sub dbReadMeta {
  my ($self, $databaseName, $metaKey, $skipCommit) = @_;

  # pass 1 to use string keys for meta properties
  return $self->dbReadOne($databaseName . $metaDbNamePart, $metaKey, $skipCommit, 1);
}

#@param <String> $databaseName : whatever the user wishes to prefix the meta name with
#@param <String> $metaKey : this is our "position" in the meta database
 # a.k.a the top-level key in that meta database, what type of meta data this is
#@param <HashRef|Scalar> $data : {someField => someValue} or a scalar value
sub dbPatchMeta {
  my ( $self, $databaseName, $metaKey, $data ) = @_;

  my $dbName = $databaseName . $metaDbNamePart;
  # If the user treats this metaKey as a scalar value, overwrite whatever was there
  if(!ref $data) {
    # undef : commit every transcation
    # 1 : use string keys
    $self->dbPut($dbName, $metaKey, $data, undef, 1);
  } else {
    # Pass 1 to merge $data with whatever was kept at this metaKey
    # Pass 1 to use string keys for meta databases
    $self->dbPatchHash($dbName, $metaKey, $data, undef, undef, 1, 1);
  }

  # Make sure that we update/sync the meta data asap, since this is critical
  # to db integrity
  $self->dbForceCommit($dbName);
  return;
}

sub dbDeleteMeta {
  my ( $self, $databaseName, $metaKey ) = @_;

  #dbDelete returns nothing
  # last argument means non-integer keys
  $self->dbDelete($databaseName . $metaDbNamePart, $metaKey, 1);
  return;
}

sub dbDropDatabase {
  my ( $self, $chr, $remove, $stringKeys) = @_;

  #dbDelete returns nothing
  # 0 means don't create
  # last argument means non-integer keys
  my $db = $self->_getDbi($chr, 0, $stringKeys);
  if(!$db->{db}->Alive) {
    $db->{db}->Txn = $db->{env}->BeginTxn();
    # not strictly necessary, but I am concerned about hard to trace abort bugs related to scope
    $db->{db}->Txn->AutoCommit(1);
  }

  # if $remove is not truthy, database is emptied rather than dropped
  $db->{db}->drop($remove);

  $instanceConfig{databaseDir}->child($chr)->remove_tree();
}

sub _getDbi {
  # Exists and not defined, because in read only database we may discover
  # that some chromosomes don't have any data (example: hg38 refSeq chrM)
  if ($envs{$_[1]}) {
    return $envs{$_[1]};
  }

  #   $_[0]  $_[1], $_[2]
  # Don't create used by dbGetNumberOfEntries
  my ($self, $name, $dontCreate, $stringKeys) = @_;

  my $dbPath = $instanceConfig{databaseDir}->child($name);

  # Create the database, only if that is what is intended
  if(!$dbPath->is_dir) {
    # If dbReadOnly flag set, this database will NEVER be created during the
    # current execution cycle
    if($instanceConfig{readOnly}) {
      return;
    } elsif ($dontCreate) {
      # dontCreate does not imply the database will never be created,
      # so we don't want to update $self->_envs
      return;
    } else {
      $dbPath->mkpath;
    }
  }

  $dbPath = $dbPath->stringify;

  my $flags;
  if($instanceConfig{readOnly}) {
    $flags = MDB_NOLOCK | MDB_NOSYNC | MDB_RDONLY | MDB_NORDAHEAD;
  } else {
    # We read synchronously during building, which is our only mixed workload
    # TODO: allow caller to configure
    $flags = MDB_NOSYNC;
  }

  my $env = LMDB::Env->new($dbPath, {
    mapsize => 128 * 1024 * 1024 * 1024, # Plenty space, don't worry
    #maxdbs => 20, # Some databases
    mode   => 0600,
    #can't just use ternary that outputs 0 if not read only...
    #MDB_RDONLY can also be set per-transcation; it's just not mentioned
    #in the docs
    flags => $flags,
    maxdbs => 0, # Some databases; else we get a MDB_DBS_FULL error (max db limit reached)
    maxreaders => 128,
  });

  if(! $env ) {
    $self->_errorWithCleanup("Failed to create environment $name for $instanceConfig{databaseDir} beacuse of $LMDB_File::last_err");
    return;
  }

  my $txn = $env->BeginTxn();

  my $dbFlags;

  # Much faster random, somewhat faster sequential performance
  # Much smaller database size (4 byte keys, vs 6-10 byte keys)
  if(!$stringKeys) {
    $dbFlags = MDB_INTEGERKEY;
  }

  my $DB = $txn->OpenDB(undef, $dbFlags);

  # ReadMode 1 gives memory pointer for perf reasons, not safe
  $DB->ReadMode(1);

  if($LMDB_File::last_err) {
    $self->_errorWithCleanup("Failed to open database $name for $instanceConfig{databaseDir} beacuse of $LMDB_File::last_err");
    return;
  }

  # Now db is open
  my $err = $txn->commit();

  if($err) {
    $self->_errorWithCleanup("Failed to commit open db tx because: $err");
    return;
  }

  $envs{$name} = {env => $env, dbi => $DB->dbi, db => $DB, tflags => $flags};

  return $envs{$name};
}

sub dbForceCommit {
  my ($self, $envName, $noSync) = @_;

  if(defined $envs{$envName}) {
    if($envs{$envName}{db}->Alive) {
      $envs{$envName}{db}->Txn->commit();
    }

    # Sync in case MDB_NOSYNC, MDB_MAPASYNC, or MDB_NOMETASYNC were enabled
    # I assume that if the user is forcing commit, they also want the state of the
    # db updated
    # sync(1) flag needed to ensure that disk buffer is flushed with MDB_NOSYNC, MAPASYNC
    $envs{$envName}{env}->sync(1) unless $noSync;
  } else {
    $self->_errorWithCleanup('dbManager expects existing environment in dbForceCommit');
  }
}

# This can be called without instantiating Seq::DBManager, either as :: or -> class method
# @param <Seq::DBManager> $self (optional)
# @param <String> $envName (optional) : the name of a specific environment
sub cleanUp {
  if(!%envs && !%cursors) {
    return 0;
  }

  if(!%envs && %cursors) {
    _fatalError('dbManager expects no cursors if no environments opened');

    return 255;
  }

  # We track the unsafe stuff, just as a precaution
  foreach (keys %cursors) {
    # Check defined because database may be empty (and will be stored as undef)
    if(defined $cursors{$_} ) {

      $cursors{$_}[1]->close();
      $cursors{$_}[0]->commit();

      delete $cursors{$_};

      if ($LMDB_File::last_err && $LMDB_File::last_err != MDB_NOTFOUND && $LMDB_File::last_err != MDB_KEYEXIST) {
        _fatalError("dbCleanUp LMDB error: $LMDB_File::last_err");

        return 255;
      }
    }
  }

  foreach (keys %envs) {
    # Check defined because database may be empty (and will be stored as undef)
    if(defined $envs{$_} ) {
      if(defined $envs{$_}{db} && $envs{$_}{db}->Alive) {
        $envs{$_}{db}->Txn->commit();
      }

      if(defined $envs{$_}{env}) {
        # Sync in case MDB_NOSYNC, MDB_MAPASYNC, or MDB_NOMETASYNC were enabled
        # sync(1) flag needed to ensure that disk buffer is flushed with MDB_NOSYNC, MAPASYNC
        $envs{$_}{env}->sync(1);
        $envs{$_}{env}->Clean();
      }

      delete $envs{$_};

      if ($LMDB_File::last_err && $LMDB_File::last_err != MDB_NOTFOUND && $LMDB_File::last_err != MDB_KEYEXIST) {
        _fatalError("dbCleanUp LMDB error: $LMDB_File::last_err");

        return 255;
      }
    }
  }

  return 0;
}

# Like DESTROY, but Moosier
sub DEMOLISH {
  my $self = shift;
  $self->cleanUp();
}

# For now, we'll throw the error, until program is changed to expect error/success
# status from functions
sub _errorWithCleanup {
  my $msg = @_ == 2 ? $_[1] : $_[0];

  cleanUp();

  _fatalError($msg);
}

sub _fatalError {
  my $msg = @_ == 2 ? $_[1] : $_[0];

  $internalLog->ERR($msg);

  # Reset error message, not sure if this is the best way
  $LMDB_File::last_err = 0;

  __PACKAGE__->log('fatal', $msg);
  die $msg;
}

1;