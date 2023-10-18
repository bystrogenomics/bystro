use 5.10.0;
use strict;
use warnings;

package MockBuilder;
use Mouse;
extends 'Seq::Base';

1;

use Test::More;
use lib 't/lib';
use TestUtils qw/ UpdateConfigAttrs /;

use Path::Tiny   qw/path/;
use Scalar::Util qw/looks_like_number/;
use YAML::XS     qw/DumpFile/;

use Seq::Tracks::Gene::Site::SiteTypeMap;
use Seq::Tracks::Reference::MapBases;

# create temp directories
my $temp_dir_db   = Path::Tiny->tempdir();
my $temp_dir_temp = Path::Tiny->tempdir();

# update config to include temp directories
my $test_config = UpdateConfigAttrs(
  './t/tracks/gene/region.yml',
  {
    database_dir => $temp_dir_db->stringify,
    temp_dir     => $temp_dir_db->stringify,
  }
);

# write new test config to file
my $test_config_file = Path::Tiny->tempfile();
DumpFile( $test_config_file, $test_config );

my $baseMapper = Seq::Tracks::Reference::MapBases->new();
my $siteTypes  = Seq::Tracks::Gene::Site::SiteTypeMap->new();

# Defines three tracks, a nearest gene , a nearest tss, and a region track
# The region track is simply a nearest track for which we storeOverlap and do not storeNearest
# To show what happens when multiple transcripts (as in NR_FAKE3, NR_FAKE3B, NR_FAKE3C)
# all share 100% of their data, except have different txEnd's, which could reveal issues with our uniqueness algorithm
# such as calculating the maximum range of the overlap: in previous code iterations
# we removed the non-unique overlapping data, without first looking at the txEnd
# and therefore had a smaller-than-expected maximum range
my $seq =
  MockBuilder->new_with_config( { config => $test_config_file, debug => 1 } );
my $tracks = $seq->tracksObj;

my $dbPath = path( $seq->database_dir );
$dbPath->remove_tree( { keep_root => 1 } );

my $refBuilder  = $tracks->getRefTrackBuilder();
my $geneBuilder = $tracks->getTrackBuilderByName('refSeq');

$refBuilder->buildTrack();
$geneBuilder->buildTrack();

my $refGetter  = $tracks->getRefTrackGetter();
my $geneGetter = $tracks->getTrackGetterByName('refSeq');

my $db = Seq::DBManager->new();

my $mainDbAref     = $db->dbReadAll('chr19');
my $regionDataAref = $db->dbReadAll( 'refSeq/chr19', 0, 1 );

# What we expect to be found in the region db
# Enough to precisely describe the tx
my @coordinateFields = (
  'chrom',  'txStart',    'txEnd',    'cdsStart',
  'cdsEnd', 'exonStarts', 'exonEnds', 'strand'
);

for my $regionEntry ( values @$regionDataAref ) {
  for my $f (@coordinateFields) {
    my $idx = $geneGetter->getFieldDbName($f);
    ok( exists $regionEntry->{$idx}, "Expected $f to exist at index $idx" );
  }

  for my $f ( @{ $geneGetter->features } ) {
    my $idx = $geneGetter->getFieldDbName($f);

    ok( exists $regionEntry->{$idx}, "Expected $f to exist at index $idx" );
  }
}

$dbPath->remove_tree( { keep_root => 1 } );
done_testing();
