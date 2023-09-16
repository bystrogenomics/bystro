use 5.10.0;
use strict;
use warnings;

package MockBuilder;
use lib './lib';
use Mouse;
extends 'Seq::Base';

1;

use Test::More;
use Path::Tiny   qw/path/;
use Scalar::Util qw/looks_like_number/;
use YAML::XS     qw/LoadFile/;
use DDP;
use Seq::Tracks::Gene::Site::SiteTypeMap;
use Seq::Tracks::Reference::MapBases;
use Seq::DBManager;

my $baseMapper  = Seq::Tracks::Reference::MapBases->new();
my $siteTypeMap = Seq::Tracks::Gene::Site::SiteTypeMap->new();

# Defines three tracks, a nearest gene , a nearest tss, and a region track
# The region track is simply a nearest track for which we storeOverlap and do not storeNearest
# To show what happens when multiple transcripts (as in NR_FAKE3, NR_FAKE3B, NR_FAKE3C)
# all share 100% of their data, except have different txEnd's, which could reveal issues with our uniqueness algorithm
# such as calculating the maximum range of the overlap: in previous code iterations
# we removed the non-unique overlapping data, without first looking at the txEnd
# and therefore had a smaller-than-expected maximum range
my $seq = MockBuilder->new_with_config(
  { config => './t/tracks/gene/ncrna.yml', debug => 1 } );
my $tracks = $seq->tracksObj;

my $dbPath = path( $seq->database_dir );
$dbPath->remove_tree( { keep_root => 1 } );

###### First make fake reference track, using the column of sequence data #####

my $refBuilder = $tracks->getRefTrackBuilder();
my $refIdx     = $refBuilder->dbName;

my @pos = ( 60950 .. 70966 );

my $file     = $seq->getReadFh( $refBuilder->local_files->[0] );
my @sequence = <$file>;
chomp @sequence;

if ( @sequence != @pos ) {
  die "malformed test, sequence != pos array";
}

my $db = Seq::DBManager->new();

for my $idx ( 0 .. $#pos ) {
  if ( !defined $baseMapper->baseMap->{ $sequence[$idx] } ) {
    die "malformed test, reference base mapper broken for base $sequence[$idx]";
  }

  my $base = $baseMapper->baseMap->{ $sequence[$idx] };

  $db->dbPatch( 'chr19', $refIdx, $pos[$idx], $base );
}

for my $idx ( 0 .. $#pos ) {
  # my $base = $baseMapper->baseMap->{$sequence[$idx]};
  my $result = $db->dbReadOne( 'chr19', $pos[$idx] );

  my $base = $baseMapper->baseMapInverse->[ $result->[$refIdx] ];

  if ( $base ne $sequence[$idx] ) {
    die "malformed test, creating fake ref track didn't work";
  }
}

my $geneBuilder = $tracks->getTrackBuilderByName('refSeq');

my $refGetter  = $tracks->getRefTrackGetter();
my $geneGetter = $tracks->getTrackGetterByName('refSeq');

my $siteTypeDbName = $geneGetter->getFieldDbName('siteType');
my $funcDbName     = $geneGetter->getFieldDbName('exonicAlleleFunction');
# my $funcDbName = $geneGetter->getFieldDbName('exonicAlleleFunction');
# my $funcDbName = $geneGetter->getFieldDbName('exonicAlleleFunction');
# my $funcDbName = $geneGetter->getFieldDbName('exonicAlleleFunction');

$geneBuilder->buildTrack();

### We have:

my $regionDataAref = $db->dbReadAll('refSeq/chr19');

my $geneIdx = $geneBuilder->dbName;
my $header  = Seq::Headers->new();

my $features = $header->getParentFeatures('refSeq');

my ( $siteTypeIdx, $funcIdx );

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
}

# Safe for use when instantiated to static variable; no set - able properties
my $coding         = $siteTypeMap->codingSiteType;
my $utr5           = $siteTypeMap->fivePrimeSiteType;
my $utr3           = $siteTypeMap->threePrimeSiteType;
my $spliceAcceptor = $siteTypeMap->spliceAcSiteType;
my $spliceDonor    = $siteTypeMap->spliceDonSiteType;
my $ncRNA          = $siteTypeMap->ncRNAsiteType;
my $intronic       = $siteTypeMap->intronicSiteType;

#                   txStart  txEnd   cdsStart   cdsEnd    exonStarts          exonEnds
# NR_033266 chr19 - 60950    70966   70966      70966  3  60950,66345,70927,  61894,66499,70966,
for my $pos ( 0 .. 100000 ) {
  my $mainDbAref = $db->dbReadOne( 'chr19', $pos );

  # 70966 is +1 of the transcript (txEnd is open)
  # and 60950 is +0 of the transcript (txStart is closed)
  # anything outside of that is missing/intergenic
  if ( $pos < 60950 || $pos > 70965 ) {
    # $intergenic++;
    ok( !defined $mainDbAref->[$geneIdx] );
    next;
  }

  # $genic++;
  ok( defined $mainDbAref->[$geneIdx] );

  my $refBase = $refGetter->get($mainDbAref);
  my $alt     = 'A';

  my $out        = [];
  my $refSeqData = $geneGetter->get( $mainDbAref, 'chr19', $refBase, $alt, 0, $out );

  # non-coding transcripts don't have UTR3/5 (not translated)
  # exonEnds closed, show this explicitly
  if ( $pos >= 60950 && $pos < 61894 ) {
    ok( $out->[$siteTypeIdx][0] eq $ncRNA );
  }

  if ( $pos >= 61894 && $pos < 66345 ) {
    if ( $pos == 61894 || $pos == 61895 ) {
      # we're on the negative strand, so should be acceptor
      ok( $out->[$siteTypeIdx][0] eq $spliceAcceptor );
      next;
    }

    if ( $pos == 66343 || $pos == 66344 ) {
      # we're on the negative strand, so should be donor at "end"
      ok( $out->[$siteTypeIdx][0] eq $spliceDonor );
      next;
    }

    # we're on the negative strand, so should be donor at "end"
    ok( $out->[$siteTypeIdx][0] eq $intronic );
  }

  if ( $pos >= 66345 && $pos < 66499 ) {
    ok( $out->[$siteTypeIdx][0] eq $ncRNA );
  }

  # before 3rd exon
  # 66499 is exonEnds of exon 2
  # 70927 is exonStarts of exon 3
  if ( $pos >= 66499 && $pos < 70927 ) {
    if ( $pos == 66499 || $pos == 66500 ) {
      # negative strand
      ok( $out->[$siteTypeIdx][0] eq $spliceAcceptor );
      next;
    }

    if ( $pos == 70925 || $pos == 70926 ) {
      # negative strand
      ok( $out->[$siteTypeIdx][0] eq $spliceDonor );
      next;
    }

    ok( $out->[$siteTypeIdx][0] eq $intronic );
  }

  #3rd exon
  if ( $pos >= 70927 && $pos < 70966 ) {
    ok( $out->[$siteTypeIdx][0] eq $ncRNA );
  }

}

# ok($inGeneCount == $hasGeneCount, "We have a refSeq record for every position from txStart to txEnd");

$db->cleanUp();
$dbPath->remove_tree( { keep_root => 1 } );

done_testing();
