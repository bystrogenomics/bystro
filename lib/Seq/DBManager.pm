use 5.10.0;
use strict;
use warnings;

package Seq::DBManager;

our $VERSION = '0.001';

# ABSTRACT: Manages Database connection
# VERSION

#TODO: Better errors; Seem to get bad perf if copy error after each db call
#TODO: Allow missing database only if in $dbReadOnly mode

use Mouse 2;
with 'Seq::Role::Message';

use Data::MessagePack;
use LMDB_File qw(:all);
use Types::Path::Tiny qw/AbsPath/;
use Sort::XS;
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

#has database_dir => (is => 'ro', isa => 'Str', required => 1, default => sub {$databaseDir});

# Instance variable holding our databases; this way can be used
# in environment where calling process never dies
# {
#   database_dir => {
#     env => $somEnv, dbi => $someDbi
#   }
# }
####################### Static Properties ############################
state $databaseDir;
# Each process should open each environment only once.
# http://www.lmdb.tech/doc/starting.html
state $envs = {};
# Read only state is shared across all instances. Lock-less reads are dangerous
state $dbReadOnly;

# Can call as class method (DBManager->setDefaultDatabaseDir), or as instance method
sub setGlobalDatabaseDir {
  my $wantedDir = @_ == 2 ? $_[1] : $_[0];

  if(defined $databaseDir) {
    $internalLog->WARN("setGlobalDatabaseDir called again, but in different environemnt than previously".
    " Before: $databaseDir, now: $wantedDir. Keeping $databaseDir");

    return;
  }

  $databaseDir //= $wantedDir;
}

# Can call as class method (DBManager->setReadOnly), or as instance method
sub setReadOnly {
  my $wanted = @_ == 2 ? $_[1] : $_[0];;

  if(defined $dbReadOnly && $dbReadOnly > $wanted && defined $envs && %$envs) {
    $internalLog->WARN("Environments open: Cannot promote from read only mode to read/write mode");

    return;
  }

  $dbReadOnly //= @_ == 2 ? $_[1] : $_[0];
}

# Prepares the class for consumption; should be run before the program can fork
# To ensure that all old data is cleared, if executing from a long-running process
sub initialize {
  cleanUp();

  $databaseDir = undef;
  $dbReadOnly = undef;
}

sub BUILD {
  my $self = shift;

  # TODO: think about better way to initialize this class w.r.t databaseDir
  if(!$databaseDir) {
    $self->_errorWithCleanup("DBManager requires databaseDir");
  }

  my $dbDir = path($databaseDir)->absolute();

  if(!$dbDir->exists) { $dbDir->mkpath; }
  if(!$dbDir->is_dir) { $self->_errorWithCleanup('database_dir not a directory'); }
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
  my $db = $_[0]->_getDbi($_[1], 1, $_[4]) or return undef;

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

  return $json ? $mp->unpack($json) : undef;
}

# Unsafe for $_[2] ; will be modified if an array is passed
# Read transactions are committed by default
sub dbRead {
  #my ($self, $chr, $posAref, $skipCommit, $stringKeys) = @_;
  #== $_[0], $_[1], $_[2],    $_[3],       $_[4] (don't assign to avoid copy)
  if(!ref $_[2]) {
    goto &dbReadOne;
  }

  #It is possible not to find a database in $dbReadOnly mode (for ex: refSeq for a while didn't have chrM)
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

  # Modifies $_[2] aka $posAref to avoid extra allocation
  for my $pos (@{ $_[2] }) {
    $txn->get($dbi, $pos, $json);

    $pos = $json ? $mp->unpack($json) : undef;
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
  my $db = $_[0]->_getDbi($_[1], 0, $_[7]);
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

#cursor version
# Read transactions are by default not committed
sub dbReadAll {
  my ( $self, $chr, $skipCommit, $stringKeys) = @_;
  #==   $_[0], $_[1], $_[2]

  #It is possible not to find a database in $dbReadOnly mode (for ex: refSeq for a while didn't have chrM)
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

################################################################################
###### For performance reasons we may want to manage our own transactions ######
######################## WARNING: *UNSAFE* #####################################
sub dbStartCursorTxn {
  #my ( $self, $chr ) = @_;

  #It is possible not to find a database in $dbReadOnly mode (for ex: refSeq for a while didn't have chrM)
  #http://ideone.com/uzpdZ8

  #        $self->_getDbi($chr)
  my $db = $_[0]->_getDbi($_[1]) or return;

  # TODO: Investigate why a subtransaction isn't successfully made
  # when using BeginTxn()
  # If we create a txn and assign it to DB->Txn, from $db->{env}, before creating a txn here
  # upon trying to use the parent transaction, we will get a crash (-30782 / BAD_TXN)
  # no such issue arises the other way around; i.e creating this transaction, then having
  # a normal DB->Txn created as a nested transaction
  if($db->{db}->Alive) {
    $_[0]->_errorWithCleanup("DB alive when calling dbStartCursorTxn; LMDB_File requires to be committed; commit DB->Txn before dbStartCursorTxn");
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
    $_[0]->_errorWithCleanup("Couldn't open cursor for $_[1]");
    return 255;
  }

  # Unsafe, private LMDB_File method access but Cursor::open does not track cursors
  $LMDB::Txn::Txns{$$txn}{Cursors}{$$cursor} = 1;

  # We store data in sequential, integer order
  # in all but the meta tables, which don't use this function
  # LMDB::Cursor::open($txn, $db->{dbi}, my $cursor);
  return [$txn, $cursor];
}

# Assumes user manages their own transactions
sub dbReadOneCursorUnsafe {
  #my ($self, $cursor, $pos) = @_;
      #$_[0]. $_[1].   $_[2]

  $_[1]->[1]->_get($_[2], my $json, MDB_SET);

  if($LMDB_File::last_err) {
    if($LMDB_File::last_err != MDB_NOTFOUND) {
    #$self
      $_[0]->_errorWithCleanup("dbEndCursorTxn LMDB error: $LMDB_File::last_err");
      return 255;
    }

    #reset the class error variable, to avoid crazy error reporting later
    $LMDB_File::last_err = 0;
  }

  return $json ? $mp->unpack($json) : undef;
}

sub dbReadCursorUnsafe {
  #my ($self, $cursor, $posAref) = @_;
      #$_[0]. $_[1].   $_[2]

  foreach (@{$_[2]}) {
    $_[1]->[1]->_get($_, my $json, MDB_SET);

    $_ = $json ? $mp->unpack($json) : undef;
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
    if($LMDB_File::last_err != MDB_NOTFOUND) {
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
  # my ( $self, $cursor, $chr ) = @_;
  #.     $_[0], $_[1],   $_[2]

  #        $self->_getDbi($chr)
  # my $db = $_[0]->_getDbi($_[2]) or return undef;
  #$cursor->close();
  $_[1]->[1]->close();

  # closes a write cursor as well; the above $cursor->close() is to be explicit
  # will not close a MDB_RDONLY cursor
  $_[1]->[0]->commit();

  # get rid of this object; we want to help user not try to re-use it
  # TODO: only in case of non-MDB_RDONLY
  undef $_[1];

  if($LMDB_File::last_err) {
    if($LMDB_File::last_err != MDB_NOTFOUND) {
     #$self->
      $_[0]->_errorWithCleanup("dbEndCursorTxn LMDB error: $LMDB_File::last_err");
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

#to store any records
#For instance, here we can store our feature name mappings, our type mappings
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

  path($databaseDir)->child($chr)->remove_tree();
}

sub _getDbi {
  # Exists and not defined, because in read only database we may discover
  # that some chromosomes don't have any data (example: hg38 refSeq chrM)
  if ($envs->{$_[1]}) {
    return $envs->{$_[1]};
  }

  #   $_[0]  $_[1], $_[2]
  # Don't create used by dbGetNumberOfEntries
  my ($self, $name, $dontCreate, $stringKeys) = @_;

  my $dbPath = path($databaseDir)->child($name);

  # Create the database, only if that is what is intended
  if(!$dbPath->is_dir) {
    # If dbReadOnly flag set, this database will NEVER be created during the
    # current execution cycle
    if($dbReadOnly) {
      $envs->{$name} = undef;
      return $envs->{$name};
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
  if($dbReadOnly) {
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
  });

  if(! $env ) {
    $self->_errorWithCleanup("Failed to create environment $name for $databaseDir beacuse of $LMDB_File::last_err");
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
    $self->_errorWithCleanup("Failed to open database $name for $databaseDir beacuse of $LMDB_File::last_err");
    return;
  }

  # Now db is open
  my $err = $txn->commit();

  if($err) {
    $self->_errorWithCleanup("Failed to commit open db tx because: $err");
    return;
  }

  $envs->{$name} = {env => $env, dbi => $DB->dbi, db => $DB, tflags => $flags};

  return $envs->{$name};
}

sub dbForceCommit {
  my ($self, $envName, $noSync) = @_;

  if(defined $envs->{$envName}) {
    if($envs->{$envName}{db}->Alive) {
      $envs->{$envName}{db}->Txn->commit();
    }

    # Sync in case MDB_NOSYNC, MDB_MAPASYNC, or MDB_NOMETASYNC were enabled
    # I assume that if the user is forcing commit, they also want the state of the
    # db updated
    # sync(1) flag needed to ensure that disk buffer is flushed with MDB_NOSYNC, MAPASYNC
    $envs->{$envName}{env}->sync(1) unless $noSync;
  } else {
    $self->_errorWithCleanup('dbManager expects existing environment in dbForceCommit');
  }
}

# This can be called without instantiating Seq::DBManager;
# @param <Seq::DBManager> $self (optional)
# @param <String> $envName (optional) : the name of a specific environment
sub cleanUp {
  #$envName is typically the chromosome
  my ($self, $envName) = @_;

  if(!%$envs) {
    return;
  }

  foreach (defined $envName ? $envName : keys %$envs) {
    # Check defined because database may be empty (and will be stored as undef)
    if(defined $envs->{$_} ) {
      if($envs->{$_}{db}->Alive) {
        $envs->{$_}{db}->Txn->commit();
      }

      # Sync in case MDB_NOSYNC, MDB_MAPASYNC, or MDB_NOMETASYNC were enabled
      # sync(1) flag needed to ensure that disk buffer is flushed with MDB_NOSYNC, MAPASYNC
      $envs->{$_}{env}->sync(1);
      $envs->{$_}{env}->Clean();

      delete $envs->{$_};
    }
  }
}

# For now, we'll throw the error, until program is changed to expect error/success
# status from functions
sub _errorWithCleanup {
  my ($self, $msg) = @_;

  cleanUp();
  $internalLog->ERR($msg);
  # Reset error message, not sure if this is the best way
  $LMDB_File::last_err = 0;

  # Make it easier to track errors
  say STDERR "LMDB error: $msg";

  $self->log('error', $msg);
  exit(255);
}

__PACKAGE__->meta->make_immutable;

1;