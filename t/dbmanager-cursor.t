use 5.10.0;
use strict;
use warnings;

package TestMe;
use Test::More;
use lib './lib';
use Seq::Tracks::Build;
use Try::Tiny;
use DDP;

system('rm -rf ./t/db/index');

Seq::DBManager::initialize({
  databaseDir =>'./t/db/index'
});

my $db = Seq::DBManager->new();

my $dbIdx = 1;
my $pos = 99;
my $val = "HELLO WORLD";

ok(!%LMDB::Env::Envs, "prior to first transaction, no transactions listed");

### Test Unsafe Transactions (Manually Managed) ##########
my $cursor = $db->dbStartCursorTxn('test');

ok(keys %LMDB::Env::Envs == 1, "after opening cursors, have one transaction");

my ($expected, $err);

ok(ref $cursor && ref $cursor eq 'ARRAY', 'dbStartCursorTxn returns an array');
ok(ref $cursor->[0] eq 'LMDB::Txn', 'dbStartCursorTxn first return item is an LMDB::Txn');
ok(ref $cursor->[1] eq 'LMDB::Cursor', 'dbStartCursorTxn 2nd return item is an LMDB::Cursor');

$err = $db->dbPatchCursorUnsafe($cursor, 'test', $dbIdx, $pos, $val);

ok($err == 0, "dbPatchCursorUnsafe returns 0 error status upon successful insertion");

$expected = $db->dbReadOneCursorUnsafe($cursor, $pos);

ok(defined $expected, "Before committing, we can see inserted value, as we have stayed within a single transaction");
ok($#$expected == $dbIdx && !defined $expected->[0] && $expected->[1] eq $val, "dbReadCursorUnsafe returns an array of track data; each index is another track");

$err = $db->dbEndCursorTxn('test');

$expected = $db->dbReadOne('test', $pos);

ok(defined $expected, "After committing, we can see inserted value using dbReadOne w/ commit");
ok($#$expected == $dbIdx && !defined $expected->[0] && $expected->[1] eq $val, "dbReadOne returns an array of track data; each index is another track");

$expected = $db->dbReadOne('test', $pos, 1);

ok($expected->[1] eq $val, "After committing, we can see inserted value using dbReadOne w/o commit");

my $cursorErr;
try {
  $cursor = $db->dbStartCursorTxn('test');
} catch {
  $cursorErr = $_;
};

ok(defined $cursorErr, "Cannot open cursor transaction while active transaction for the given dbi");

my $commitErr;
try {
  $db->dbForceCommit('test', 1);
} catch {
  $commitErr = $_;
};

ok(defined $commitErr && $commitErr =~ /expects existing environment/, "Fatal errors clear dbManager environment state");
# ok(defined $commitErr, "dbForceCommit is a void function");

# Note, unfortunately we can do this, 
$cursor = $db->dbStartCursorTxn('test');

ok(defined $cursor, "dbForceCommit w/o force sync successfully closes the DB associated txn, allowing us to create a new transaction");

$expected = $db->dbReadOne('test', $pos, 1);

ok($#$expected == $dbIdx && !defined $expected->[0] && $expected->[1] eq $val, "Can use dbReadOne, without committing, as subtransaction of cursor-containing separate transaction");

$err = $db->dbPatchCursorUnsafe($cursor, 'test', $dbIdx, $pos, "SOMETHING NEW");

ok($err == 0, "Can run dbPatchCursorUnsafe, with uncommitted child transaction");

ok($expected->[$dbIdx] eq 'HELLO WORLD', "we don't overwrite entries");

$err = $db->dbEndCursorTxn('test');

ok($err == 0, 'dbEndCursorTxn returns 0 upon no error');

done_testing();