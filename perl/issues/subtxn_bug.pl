use 5.10.0;
use strict;
use warnings;

package DBManager;
use LMDB_File qw/:all/;
my %envs;

sub new {
  my $class = shift;
  my $self  = {};
  bless $self, $class;
}

sub dbPut {
  my ( $self, $dbName, $key, $data, $skipCommit ) = @_;

  # 0 to create database if not found
  my $db = $self->_getDbi($dbName);

  if ( !$db->{db}->Alive ) {
    $db->{db}->Txn = $db->{env}->BeginTxn();

    # not strictly necessary, but I am concerned about hard to trace abort bugs related to scope
    $db->{db}->Txn->AutoCommit(1);
  }

  $db->{db}->Txn->put( $db->{dbi}, $key, $data );

  $db->{db}->Txn->commit() unless $skipCommit;

  if ($LMDB_File::last_err) {
    if ( $LMDB_File::last_err != MDB_KEYEXIST ) {
      die $LMDB_File::last_err;
    }

    #reset the class error variable, to avoid crazy error reporting later
    $LMDB_File::last_err = 0;
  }

  return 0;
}

sub dbReadOne {
  my ( $self, $dbName, $key, $skipCommit ) = @_;

  my $db = $self->_getDbi($dbName) or return undef;

  if ( !$db->{db}->Alive ) {
    $db->{db}->Txn = $db->{env}->BeginTxn();

    # not strictly necessary, but I am concerned about hard to trace abort bugs related to scope
    $db->{db}->Txn->AutoCommit(1);
  }

  $db->{db}->Txn->get( $db->{dbi}, $key, my $data );

  # Commit unless the user specifically asks not to
  #if(!$skipCommit) {
  $db->{db}->Txn->commit() unless $skipCommit;

  if ($LMDB_File::last_err) {
    if ( $LMDB_File::last_err != MDB_NOTFOUND ) {
      die $LMDB_File::last_err;
    }

    $LMDB_File::last_err = 0;
  }

  return $data;
}

sub dbStartCursorTxn {
  my ( $self, $dbName ) = @_;

  my $db = $self->_getDbi($dbName) or return;

  my $txn = $db->{env}->BeginTxn();

  # Help LMDB_File track our cursor
  LMDB::Cursor::open( $txn, $db->{dbi}, my $cursor );

  # Unsafe, private LMDB_File method access but Cursor::open does not track cursors
  $LMDB::Txn::Txns{$$txn}{Cursors}{$$cursor} = 1;

  return [ $txn, $cursor ];
}

sub _getDbi {

  # Exists and not defined, because in read only database we may discover
  # that some chromosomes don't have any data (example: hg38 refSeq chrM)

  #   $_[0]  $_[1], $_[2]
  # Don't create used by dbGetNumberOfEntries
  my ( $self, $dbPath ) = @_;

  if ( $envs{$dbPath} ) {
    return $envs{$dbPath};
  }

  my $env = LMDB::Env->new(
    $dbPath,
    {
      mapsize => 128 * 1024 * 1024 * 1024, # Plenty space, don't worry
      #maxdbs => 20, # Some databases
      mode   => 0600,
      maxdbs =>
        0, # Some databases; else we get a MDB_DBS_FULL error (max db limit reached)
    }
  );

  if ( !$env ) {
    die 'No env';
  }

  my $txn = $env->BeginTxn();

  my $dbFlags;

  my $DB = $txn->OpenDB( undef, MDB_INTEGERKEY );

  # ReadMode 1 gives memory pointer for perf reasons, not safe
  $DB->ReadMode(1);

  if ($LMDB_File::last_err) {
    die $LMDB_File::last_err;
  }

  # Now db is open
  my $err = $txn->commit();

  if ($err) {
    die $err;
  }

  $envs{$dbPath} = { env => $env, dbi => $DB->dbi, db => $DB };

  return $envs{$dbPath};
}

1;

use Test::More;
use DDP;
my $db = DBManager->new();

my $dbIdx = 1;
my $pos   = 99;
my $val   = "HELLO WORLD";

system('rm -rf ./test && mkdir ./test');

#### WORKS GREAT ####
my $cursor;
$cursor = $db->dbStartCursorTxn('test');

### Test Unsafe Transactions (Manually Managed) ##########
$db->dbPut( 'test', $pos, [], 1 );

$db->dbReadOne( 'test', $pos );

p %LMDB::Env::Envs;

$db->dbReadOne( 'test', $pos );
undef $db;
undef $cursor;

system('rm -rf ./test && mkdir ./test');

$db = DBManager->new();
#### DIES MISERABLE DEATH ####
say "The reverse order doesn't work";

$db->dbPut( 'test', $pos, [], 1 );
$cursor = $db->dbStartCursorTxn('test');
$db->dbReadOne( 'test', $pos );

say "We will never see this";
