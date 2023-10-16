use 5.10.0;
use strict;
use warnings;

package MockBuilder;
use Mouse;
extends 'Seq::Base';

1;

use Test::More;
use Path::Tiny   qw/path/;
use Scalar::Util qw/looks_like_number/;
use YAML::XS     qw/LoadFile/;

use Seq::Tracks::Gene::Site::SiteTypeMap;
use Seq::Tracks::Reference::MapBases;

my $baseMapper = Seq::Tracks::Reference::MapBases->new();
my $siteTypes  = Seq::Tracks::Gene::Site::SiteTypeMap->new();

# Defines three tracks, a nearest gene , a nearest tss, and a region track
# The region track is simply a nearest track for which we storeOverlap and do not storeNearest
# To show what happens when multiple transcripts (as in NR_FAKE3, NR_FAKE3B, NR_FAKE3C)
# all share 100% of their data, except have different txEnd's, which could reveal issues with our uniqueness algorithm
# such as calculating the maximum range of the overlap: in previous code iterations
# we removed the non-unique overlapping data, without first looking at the txEnd
# and therefore had a smaller-than-expected maximum range
my $seq = MockBuilder->new_with_config(
  { config => './t/tracks/gene/simple.yml', debug => 1 } );
my $tracks = $seq->tracksObj;

my $dbPath = path( $seq->database_dir );
$dbPath->remove_tree( { keep_root => 1 } );

my $refBuilder  = $tracks->getRefTrackBuilder();
my $geneBuilder = $tracks->getTrackBuilderByName('refSeq');

my $refGetter  = $tracks->getRefTrackGetter();
my $geneGetter = $tracks->getTrackGetterByName('refSeq');

$refBuilder->buildTrack();
$geneBuilder->buildTrack();

### We have:

my $db = Seq::DBManager->new();

my $mainDbAref     = $db->dbReadAll('chrM');
my $regionDataAref = $db->dbReadAll('refSeq/chrM');

my $geneDbName = $geneBuilder->dbName;
my $header     = Seq::Headers->new();

my $features = $header->getParentFeatures('refSeq');

# my $txNumberOutIdx = first_index { $_ eq $geneGetter->txNumberKey } @$features;

my ( $siteTypeIdx, $funcIdx, $txNumberOutIdx );

for ( my $i = 0; $i < @$features; $i++ ) {
  my $feat = $features->[$i];

  if ( $feat eq 'siteType' ) {
    $siteTypeIdx = $i;
    next;
  }

  if ( $feat eq 'exonicAlleleFunction' ) {
    $funcIdx = $i;
    next;
  }

  if ( $feat eq $geneGetter->txNumberKey ) {
    $txNumberOutIdx = $i;
    next;
  }
}

my $hasGeneCount = 0;
my $inGeneCount  = 0;

for my $pos ( 0 .. $#$mainDbAref ) {
  my $dbData = $mainDbAref->[$pos];

  if ( $pos >= 1672 && $pos < 3230 ) {
    $inGeneCount++;

    my @out;

    # not an indel
    my $posIdx = 0;
    $geneGetter->get( $dbData, 'chrM', $refGetter->get($dbData), 'A', $posIdx, \@out );

    ok( join( ",", @{ $out[$siteTypeIdx] } ) eq $siteTypes->ncRNAsiteType,
      'ncRNA site type' );
    ok( !defined $out[$funcIdx][0],    'ncRNA has no exonicAlleleFunction' );
    ok( $out[$txNumberOutIdx][0] == 0, 'txNumber is outputted if requested' );
  }

  if ( defined $dbData->[$geneDbName] ) {
    $hasGeneCount++;
  }

  my @out;
  my $refSite = $refGetter->get($dbData);
}

ok( $inGeneCount == $hasGeneCount,
  "We have a refSeq record for every position from txStart to txEnd" );

$db->cleanUp();
$dbPath->remove_tree( { keep_root => 1 } );

done_testing();
