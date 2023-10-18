use 5.10.0;
use strict;
use warnings;

use Test::More;

use Seq::Tracks::Build;

# create temp directories
my $db_dir    = Path::Tiny->tempdir();
my $files_dir = Path::Tiny->tempdir();

Seq::DBManager::initialize( { databaseDir => $db_dir } );

my $t = Seq::Tracks::Build->new(
  {
    files_dir   => $files_dir,
    name        => 'test',
    type        => 'sparse',
    chromosomes => [ 'chr1', '1', 'chr2', '2', '0' ],
    assembly    => 'hgTest'
  }
);

my $wantedChr = $t->chrWantedAndIncomplete('chr1');

ok( $wantedChr eq 'chr1' );

$wantedChr = $t->chrWantedAndIncomplete('chr2');

ok( $wantedChr eq 'chr2' );

$wantedChr = $t->chrWantedAndIncomplete('chr3');

ok( !defined $wantedChr, "Unwanted chromsomes result in undef returned" );

$wantedChr = $t->chrWantedAndIncomplete('1');

ok( $wantedChr eq '1' );

$wantedChr = $t->chrWantedAndIncomplete('0');

ok( $wantedChr eq '0', "'0' accepted as a valid chromosome number" );

$wantedChr = $t->chrWantedAndIncomplete('');

ok( !defined $wantedChr,
  "Empty strings not accepted as valid chromosome, result in undef returned" );

done_testing();
