use 5.10.0;
use strict;
use warnings;

package TestMe;
use Test::More;
use Seq::Tracks::Build;
use Try::Tiny;
use DDP;
use Data::MessagePack;

my $mp = Data::MessagePack->new();

system('rm -rf ./t/db/index-del');

Seq::DBManager::initialize( { databaseDir => './t/db/index-del' } );

my $db = Seq::DBManager->new();

my $chr   = 'test';
my $dbIdx = 1;
my $pos   = 99;

my $vals = [ 0, 1, [ 2, 3 ], "hello", "world" ];

# off the end of $vals; we'll periodically add a val to end at 5th index
$dbIdx = 5;

for my $pos ( 0 .. 100 ) {
  my @v = @$vals;

  if ( $pos % 2 ) {
    push @v, "val " . $pos;
  }

  $db->dbPut( $chr, $pos, \@v );
}

for my $pos ( 0 .. 100 ) {
  my $readV = $db->dbReadOne( $chr, $pos );

  if ( $pos % 2 ) {
    ok( $readV->[5] eq "val " . $pos, "could insert and read values at end" );
    next;
  }

  ok( !defined $readV->[5], "could insert and read values at end" );
}

$db->dbDeleteAll( $chr, $dbIdx );

for my $pos ( 0 .. 100 ) {
  my $readV = $db->dbReadOne( $chr, $pos );

  ok( !defined $readV->[5], "could delete value from end" );
  ok( $readV->[4] eq 'world',
    "deleting in middle doesn\'t impact preceding adjacent value" );
  ok( $readV->[3] eq 'hello',
    "deleting in middle doesn\'t impact 2nd preceding adjacent value" );
}

system('rm -rf ./t/db/index-del');

$vals = [ 0, 1, [ 2, 3 ], undef, "end" ];

# in the middle
$dbIdx = 3;

for my $pos ( 0 .. 100 ) {
  my @v = @$vals;

  if ( $pos % 2 ) {
    $v[$dbIdx] = "val in middle " . $pos;
  }

  $db->dbPut( $chr, $pos, \@v );
}

for my $pos ( 0 .. 100 ) {
  my $readV = $db->dbReadOne( $chr, $pos );

  if ( $pos % 2 ) {
    ok(
      $readV->[$dbIdx] eq "val in middle " . $pos,
      "could insert and read values at middle"
    );
    next;
  }

  ok( !defined $readV->[5], "could insert and read values at middle" );
}

$db->dbDeleteAll( $chr, $dbIdx );

for my $pos ( 0 .. 100 ) {
  my $readV = $db->dbReadOne( $chr, $pos );

  ok( !defined $readV->[$dbIdx], "could delete value from middle" );
  ok(
    join( ',', @{ $readV->[2] } ) eq join( ',', 2, 3 ),
    "deleting in middle doesn\'t impact preceding adjacent value"
  );
  ok( $readV->[4] eq 'end', "deleting in middle doesn\'t impact next adjacent value" );
}

system('rm -rf ./t/db/index-del');

done_testing();
