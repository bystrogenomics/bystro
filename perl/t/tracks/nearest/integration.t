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
use DDP;

my $baseMapper = Seq::Tracks::Reference::MapBases->new();

# Defines three tracks, a nearest gene , a nearest tss, and a region track
# The region track is simply a nearest track for which we storeOverlap and do not storeNearest
# To show what happens when multiple transcripts (as in NR_FAKE3, NR_FAKE3B, NR_FAKE3C)
# all share 100% of their data, except have different txEnd's, which could reveal issues with our uniqueness algorithm
# such as calculating the maximum range of the overlap: in previous code iterations
# we removed the non-unique overlapping data, without first looking at the txEnd
# and therefore had a smaller-than-expected maximum range
my $seq = MockBuilder->new_with_config(
  { config => path('./t/tracks/nearest/test.yml')->absolute, debug => 1 } );

system( 'rm -rf ' . path( $seq->database_dir )->child('*') );

my $tracks            = $seq->tracksObj;
my $refBuilder        = $tracks->getRefTrackBuilder();
my $nearestTssBuilder = $tracks->getTrackBuilderByName('refSeq.nearestTss');
my $nearestBuilder    = $tracks->getTrackBuilderByName('refSeq.nearest');
my $geneBuilder       = $tracks->getTrackBuilderByName('refSeq.gene');

my $nearestGetter    = $tracks->getTrackGetterByName('refSeq.nearest');
my $nearestTssGetter = $tracks->getTrackGetterByName('refSeq.nearestTss');
my $geneGetter       = $tracks->getTrackGetterByName('refSeq.gene');

my $db = Seq::DBManager->new();

$refBuilder->buildTrack();
$nearestTssBuilder->buildTrack();
$nearestBuilder->buildTrack();
$geneBuilder->buildTrack();

### We have:
#name      chrom strand  txStart txEnd
#NR_137295 chrM  +        1672  3230
#NR_FAKE1  chrM  +        2000  2300
#NR_FAKE2  chrM  +        2200  3400

my $mainDbAref     = $db->dbReadAll('chrM');
my $regionDataAref = $db->dbReadAll('refSeq.nearest/chrM');

# p $mainDbAref;
my $hasNearestCount    = 0;
my $hasNearestTssCount = 0;
my $nearestDbName      = $nearestBuilder->dbName;
my $nearestTssDbName   = $nearestTssBuilder->dbName;
my $geneDbName         = $geneBuilder->dbName;

for my $dbData (@$mainDbAref) {
  if ( defined $dbData->[$nearestDbName] ) {
    $hasNearestCount++;
  }

  if ( defined $dbData->[$nearestTssDbName] ) {
    $hasNearestTssCount++;
  }
}

ok( $hasNearestCount == @$mainDbAref );
ok( $hasNearestTssCount == @$mainDbAref );

# Want to make sure that order is always preserved
my %map = (
  NR_137295 => 'RNR2',
  NR_FAKE1  => 'RNR2_FAKE1',
  NR_FAKE2  => 'RNR2_FAKE2',
  NR_FAKE3  => 'FAKE3',
  NR_FAKE3B => 'FAKE3',
  NR_FAKE3C => 'FAKE3',
  NR_FAKE4  => 'FAKE4',
);

for my $pos ( 0 .. $#$mainDbAref ) {
  my $dbData = $mainDbAref->[$pos];

  my @out;
  my $nGeneData = $nearestGetter->get( $dbData, 'chrM', 'C', 'A', 0, \@out, $pos );

  my @outTss;
  my $nTssGeneData =
    $nearestTssGetter->get( $dbData, 'chrM', 'C', 'A', 0, \@outTss, $pos );

  my @outGene;
  my $geneData = $geneGetter->get( $dbData, 'chrM', 'C', 'A', 0, \@outGene, $pos );

  # config features order is name, name2, and then we add dist in the 3rd position
  # so 0 == name 1 == name2 2 == dist
  my $name  = $out[0][0];
  my $name2 = $out[1][0];
  my $dist  = $out[2][0];

  my $nameTss  = $outTss[0][0];
  my $name2Tss = $outTss[1][0];
  my $distTss  = $outTss[2][0];

  # This track we specify should not have a distance feature in YAML
  # Only feature is name2 and we're not calculating a distnace
  # so array should have only 1 value per position entry
  ok( !@outGene || @{ $outGene[0] } == 1 );

  my $name2Gene = $outGene[0][0];

  # check that order preserved
  if ( ref $name ) {
    my $i = -1;
    for my $n (@$name) {
      $i++;
      my $expected = $map{$n};

      # If we have an array, we expect all other values (except dist)
      # to be in array form
      # ie
      # if name = [val1, val2]
      # name2 = [name2] even when there are no other unique name2 values
      # name2 = [name2_forVal1, name2_forVal2] when there is more than one unique value
      # This is done because features aren't guaranteed to be scalar
      # Let's say we have a feautre for tissues the gene is expressed in
      # for the above, when all tissues identical between transcripts, we can, without
      # loss of information, compress to
      # expression = [[kidney, spleen, pons, medulla]]
      # whereas
      # expression = [kidney, spleen, pons, medulla] would be completely incorrect
      # More detail in lib/Seq/Tracks/Nearest/Build.pm
      my $actual;
      if ( @{$name2} > 1 ) {
        $actual = $name2->[$i];
      }
      else {
        $actual = $name2->[0];
      }

      ok( $actual eq $expected );
    }
  }
  else {
    my $expected = $map{$name};
    ok( $name2 eq $expected );
  }

  if ( ref $nameTss ) {
    my $i = -1;
    for my $n (@$nameTss) {
      $i++;
      my $expected = $map{$n};

      # If we have an array, we expect all other values (except dist)
      # to be in array form
      # ie
      # if name = [val1, val2]
      # name2 = [name2] even when there are no other unique name2 values
      # name2 = [name2_forVal1, name2_forVal2] when there is more than one unique value
      # This is done because features aren't guaranteed to be scalar
      # Let's say we have a feautre for tissues the gene is expressed in
      # for the above, when all tissues identical between transcripts, we can, without
      # loss of information, compress to
      # expression = [[kidney, spleen, pons, medulla]]
      # whereas
      # expression = [kidney, spleen, pons, medulla] would be completely incorrect
      # More detail in lib/Seq/Tracks/Nearest/Build.pm
      my $actual;
      if ( @{$name2Tss} > 1 ) {
        $actual = $name2Tss->[$i];
      }
      else {
        $actual = $name2Tss->[0];
      }

      ok( $actual eq $expected );
    }
  }
  else {
    my $expected = $map{$nameTss};
    ok( $name2Tss eq $expected );
  }

  # An intergenic position
  # In all of these tests, we assume that the end position (here txEnd) is 0-based
  # This should be ensured by the nearest builder function
  if ( $pos <= 1672 ) {
    ok( $dist == 1672 - $pos );
    ok( $name eq 'NR_137295' );
    ok( $name2 eq 'RNR2' );

    # for intergenic stuff, all should be identical between
    # nearest tracks that go from .. to and those with just one endpoint (from)
    ok( $distTss == $dist );
    ok( $nameTss eq $name );
    ok( $name2Tss eq $name2 );
  }

  # 2000 here is the txStart of NR_FAKE1, so -1 that is the last non-NR_FAKE1 base
  # NR_137295 chrM  + 1672  3230  3230  3230  1 1672, 3230, 0 RNR2  NA  NA  NA  NA  NA  NA  NA  NA  NA  NA  NA
  if ( $pos >= 1672 && $pos < 2000 ) {
    ok( $dist == 0 );
    ok( $name eq 'NR_137295' );
    ok( $name2 eq 'RNR2' );

    # We always expect our gene track to provide scalars, when there is a single unique
    # value, since we can do so without any ambiguity (since 1 feature only)
    # The uniquness algorithm should do so only when that is possible
    ok( $name2Gene eq 'RNR2' );

    # for spaces between two from points of adjacent transcripts/regions
    # nearest tracks of only 'from' coordinate should treat these as essentially
    # intergenic
    # The midoint should be calcualted as txStartPrevious + txStartPrevious ... txStartNext / 2
    # since txStart is 0-based closed
    # In reality we round off, such that the actual midpoint is assigne to the downstream values
    # when we consider
    if ( $pos < 1672 + ( 2000 - 1672 ) / 2 ) {
      ok( $distTss == 1672 - $pos );
      ok( $nameTss eq 'NR_137295' );
      ok( $name2Tss eq 'RNR2' );
    }
    else {
      ok( $distTss == 2000 - $pos );
      ok( $nameTss eq 'NR_FAKE1' );
      ok( $name2Tss eq 'RNR2_FAKE1' );
    }
  }

  # 2000 is the txStart of NR_FAKE1
  # 2200 here is the txStart of NR_FAKE2, the next closest transcript when measured by txStart
  if ( $pos >= 2000 && $pos < 2200 ) {
    ok( $dist == 0 );
    ok( join( ",", @$name ) eq 'NR_137295,NR_FAKE1' );
    ok( join( ",", @$name2 ) eq 'RNR2,RNR2_FAKE1' );

    # same as nGene within the gene when storeNearest is false
    ok( join( ",", @$name2Gene ) eq 'RNR2,RNR2_FAKE1' );

    # ok(!defined $distGene);

    # for spaces between two from points of adjacent transcripts/regions
    # nearest tracks of only 'from' coordinate should treat these as essentially
    # intergenic
    my $midPoint = 2000 + ( 2200 - 2000 ) / 2;
    if ( $pos < $midPoint ) {
      ok( $distTss == 2000 - $pos );
      ok( $nameTss eq 'NR_FAKE1' );
      ok( $name2Tss eq 'RNR2_FAKE1' );
    }
    else {
      ok( $distTss == 2200 - $pos );
      ok( $nameTss eq 'NR_FAKE2' );
      ok( $name2Tss eq 'RNR2_FAKE2' );
    }
  }

  # 2300 is txEnd, so -1 for last pos in RNR2_FAKE1
  # This is the closest tx when measured by txStart .. txEnd
  # In this range, up to 3 tx overlap whn looking txStart .. txEnd
  # NR_137295 chrM  + 1672  3230  3230  3230  1 1672, 3230, 0 RNR2  NA  NA  NA  NA  NA  NA  NA  NA  NA  NA  NA
  # NR_FAKE1  chrM  + 2000  2300  3230  3230  1 1672, 3230, 0 RNR2_FAKE1  NA  NA  NA  NA  NA  NA  NA  NA  NA  NA  NA
  # NR_FAKE2  chrM  + 2200  3400  3230  3230  1 1672, 3230, 0 RNR2_FAKE2  NA  NA  NA  NA  NA  NA  NA  NA  NA  NA  NA
  if ( $pos >= 2200 && $pos < 2300 ) {
    ok( $dist == 0 );
    ok(
      join( ",", sort { $a cmp $b } @$name ) eq
        join( ",", sort { $a cmp $b } 'NR_137295', 'NR_FAKE1', 'NR_FAKE2' ) );
    ok(
      join( ",", sort { $a cmp $b } @$name2 ) eq
        join( ',', sort { $a cmp $b } 'RNR2', 'RNR2_FAKE1', 'RNR2_FAKE2' ) );

    # same as nGene within the gene when storeNearest is false
    ok(
      join( ",", sort { $a cmp $b } @$name2Gene ) eq
        join( ",", sort { $a cmp $b } 'RNR2', 'RNR2_FAKE1', 'RNR2_FAKE2' ) );

    # ok(!defined $distGene);

    #For single-point (from) nearest tracks, this case is equivalent to being
    #intergenic (or on top of @2200) and past/on top of the last transcript (NR_FAKE2)
    ok( $distTss == 2200 - $pos );
    ok( $nameTss eq 'NR_FAKE2' );
    ok( $name2Tss eq 'RNR2_FAKE2' );
  }

  # 3230 is 0-based open interval (txEnd), so -1 that is the end of RNR2
  # This is the next closest point by txEnd, and in this interval up to 2 transcripts overlap
  # when measured by txStart .. txEnd - 1
  # NR_137295 chrM  + 1672  3230  3230  3230  1 1672, 3230, 0 RNR2  NA  NA  NA  NA  NA  NA  NA  NA  NA  NA  NA
  # -> can't overlap, since ends at 2300 - 1 == 2299: NR_FAKE1  chrM  + 2000  2300  3230  3230  1 1672, 3230, 0 RNR2_FAKE1  NA  NA  NA  NA  NA  NA  NA  NA  NA  NA  NA
  # NR_FAKE2  chrM  + 2200  3400  3230  3230  1 1672, 3230, 0 RNR2_FAKE2  NA  NA  NA  NA  NA  NA  NA  NA  NA  NA  NA
  if ( $pos >= 2300 && $pos < 3230 ) {
    ok( $dist == 0 );
    ok(
      join( ",", sort { $a cmp $b } @$name ) eq
        join( ",", sort { $a cmp $b } 'NR_137295', 'NR_FAKE2' ) );
    ok(
      join( ",", sort { $a cmp $b } @$name2 ) eq
        join( ',', sort { $a cmp $b } 'RNR2', 'RNR2_FAKE2' ) );

    # We don't guarantee transcript order atm, but all features will be correct relative to
    # all transcripts
    ok(
      join( ",", sort { $a cmp $b } @$name2Gene ) eq
        join( ",", sort { $a cmp $b } 'RNR2', 'RNR2_FAKE2' ) );

    # ok(!defined $distGene);

    # For txStart, these flank 2300 - 3230
    # NR_FAKE2  chrM  + 2200  3400  3230  3230  1 1672, 3230, 0 RNR2_FAKE2  NA  NA  NA  NA  NA  NA  NA  NA  NA  NA  NA
    # NR_FAKE3  chrM  + 3800  4000  3810  3900  1 1672, 3230, 0 FAKE3 NA  NA  NA  NA  NA  NA  NA  NA  NA  NA  NA
    # NR_FAKE3B chrM  + 3800  4100  3810  3900  1 1672, 3230, 0 FAKE3 NA  NA  NA  NA  NA  NA  NA  NA  NA  NA  NA
    # NR_FAKE3C chrM  + 3800  4500  3810  3900  1 1672, 3230, 0 FAKE3 NA  NA  NA  NA  NA  NA  NA  NA  NA  NA  NA

    if ( $pos < 2200 + ( 3800 - 2200 ) / 2 ) {

      #For single-point (from) nearest tracks, this case is equivalent to being
      #intergenic (or on top of @2200) and past/on top of the last transcript (NR_FAKE2)
      ok( $distTss == 2200 - $pos );
      ok( $nameTss eq 'NR_FAKE2' );
      ok( $name2Tss eq 'RNR2_FAKE2' );
    }
    else {
      #For single-point (from) nearest tracks, this case is equivalent to being
      #intergenic (or on top of @2200) and past/on top of the last transcript (NR_FAKE2)
      ok( $distTss == 3800 - $pos );
      ok(
        join( ';', sort { $a cmp $b } @$nameTss ) eq
          join( ';', sort { $a cmp $b } 'NR_FAKE3', 'NR_FAKE3B', 'NR_FAKE3C' ) );

      # when multiple transcripts overlap AND there is at least one non-scalar value,
      # all data is represented in array form to help distinguish between multiple
      # overlapping transcripts, and one transcript with some features that contain multiple values
      # or multiple transcripts with a mix of unique and non-unique features
      ok( @$name2Tss == 1 && $name2Tss->[0] eq 'FAKE3' );
    }
  }

  # Between 3230 txStart and 3400 - 1 txEnd (since the 3400 is 0-based, open)
  # NR_137295 chrM  + 1672  3230  3230  3230  1 1672, 3230, 0 RNR2  NA  NA  NA  NA  NA  NA  NA  NA  NA  NA  NA
  # NR_FAKE1  chrM  + 2000  2300  3230  3230  1 1672, 3230, 0 RNR2_FAKE1  NA  NA  NA  NA  NA  NA  NA  NA  NA  NA  NA
  # NR_FAKE2  chrM  + 2200  3400  3230  3230  1 1672, 3230, 0 RNR2_FAKE2  NA  NA  NA  NA  NA  NA  NA  NA  NA  NA  NA
  # NR_FAKE3  chrM  + 3800  4000  3810  3900  1 1672, 3230, 0 FAKE3 NA  NA  NA  NA  NA  NA  NA  NA  NA  NA  NA
  # So we have 1672, 2000, 2200 txStarts
  # Closest to 3230 - 3400 is 2200, or 3800
  # Midpoint is 3800-2200 / 2 = +800, or 3000, so it's actually always close to NR_FAKE3
  if ( $pos >= 3230 && $pos < 3400 ) {
    ok( $dist == 0 );
    ok( $name eq 'NR_FAKE2' );
    ok( $name2 eq 'RNR2_FAKE2' );

    ok( $name2Gene eq $name2 );

    # ok(!defined $distGene);

    #For single-point (from) nearest tracks, this case is equivalent to being
    #intergenic (or on top of @2200) and past/on top of the last transcript (NR_FAKE2)
    ok( $distTss == 3800 - $pos );
    ok(
      join( ",", sort { $a cmp $b } @$nameTss ) eq
        join( ",", sort { $a cmp $b } 'NR_FAKE3', 'NR_FAKE3B', 'NR_FAKE3C' ) );
    ok( join( ",", @$name2Tss ) eq 'FAKE3' );
  }

  # Testing that if multiple transcripts share a start, but not an end, that 1) we consider the intergenic region
  # by the longest end
  # And, that we don't consider the overlap incorrectly by modifying the end to the longest end...
  # that is critical of course
  # NR_FAKE2  chrM  + 2200  3400  3230  3230  1 1672, 3230, 0 RNR2_FAKE2  NA  NA  NA  NA  NA  NA  NA  NA  NA  NA  NA
  #then ...
  # NR_FAKE3  chrM  + 3800  4000  3810  3900  1 1672, 3230, 0 FAKE3  NA  NA  NA  NA  NA  NA  NA  NA  NA  NA  NA
  # NR_FAKE3B  chrM  + 3800  4100  3810  3900  1 1672, 3230, 0 FAKE3  NA  NA  NA  NA  NA  NA  NA  NA  NA  NA  NA
  # NR_FAKE3C  chrM  + 3800  4500  3810  3900  1 1672, 3230, 0 FAKE3  NA  NA  NA  NA  NA  NA  NA  NA  NA  NA  NA
  # NR_FAKE4  chrM  + 4300  5000  3810  3900  1 1672, 3230, 0 FAKE4  NA  NA  NA  NA  NA  NA  NA  NA  NA  NA  NA
  if ( $pos >= 3400 && $pos < 3800 ) {

    # nearest is from txStart to txEnd, so use end to calc distance

    # for refSeq.gene, we don't consider intergenice
    # ok(!defined $distGene);
    # Relevant:
    # NR_FAKE2  chrM  + 2200  3400  3230  3230  1 1672, 3230, 0 RNR2_FAKE2  NA  NA  NA  NA  NA  NA  NA  NA  NA  NA  NA
    # NR_FAKE3  chrM  + 3800  4000  3810  3900  1 1672, 3230, 0 FAKE3 NA  NA  NA  NA  NA  NA  NA  NA  NA  NA  NA
    # NR_FAKE3B chrM  + 3800  4100  3810  3900  1 1672, 3230, 0 FAKE3 NA  NA  NA  NA  NA  NA  NA  NA  NA  NA  NA
    # NR_FAKE3C chrM  + 3800  4500  3810  3900  1 1672, 3230, 0 FAKE3 NA  NA  NA  NA  NA  NA  NA  NA  NA  NA  NA
    ok( !defined $name2Gene );

    if ( $pos < 3399 + ( 3800 - 3399 ) / 2 ) {
      ok( $dist = 3399 - $pos );
      ok( $name eq 'NR_FAKE2' );
      ok( $name2 eq 'RNR2_FAKE2' );
    }
    else {
      ok( $dist = 3800 - $pos );
      ok(
        join( ",", sort { $a cmp $b } @$nameTss ) eq
          join( ",", sort { $a cmp $b } 'NR_FAKE3', 'NR_FAKE3B', 'NR_FAKE3C' ) );
      ok( join( ",", @$name2 ) eq 'FAKE3' );
    }

    if ( $pos < 2200 + ( 3800 - 2200 ) / 2 ) {

      # should never appear here
      ok( $nameTss eq 'NR_FAKE2' );
      ok( $name2Tss eq 'RNR2_FAKE2' );

      # for nearestTss, only txStart is considered of NR_FAKE2
      ok( $distTss == 2200 - $pos );
    }
    else {
      ok(
        join( ",", sort { $a cmp $b } @$nameTss ) eq
          join( ",", sort { $a cmp $b } 'NR_FAKE3', 'NR_FAKE3B', 'NR_FAKE3C' ) );
      ok( join( ",", @$name2Tss ) eq 'FAKE3' );

      # for nearestTss, only txStart is considered of NR_FAKE2
      ok( $distTss == 3800 - $pos );
    }
  }

  if ( $pos >= 3800 && $pos < 5000 ) {
    if ( $pos < 4000 ) {
      ok( $dist == 0 );
      ok(
        join( ",", sort { $a cmp $b } @$name ) eq
          join( ",", sort { $a cmp $b } 'NR_FAKE3', 'NR_FAKE3B', 'NR_FAKE3C' ) );
      ok( join( ",", @$name2 ) eq 'FAKE3' );

      # name2 gene will have no array values at all here; it's equivalent of only 1 gene
      # since we're not recording any per-transcript info
      # so no array
      # TODO: this may be somewhat confusing...
      ok( $name2Gene eq 'FAKE3' );

      #For single-point (from) nearest tracks, this case is equivalent to being
      #intergenic (or on top of txStart)
      ok( $distTss == 3800 - $pos );
      ok(
        join( ",", sort { $a cmp $b } @$nameTss ) eq join( ",", sort { $a cmp $b } @$name )
      );
      ok( join( ",", @$name2Tss ) eq join( ",", @$name2 ) );
    }

    if ( $pos >= 4000 && $pos < 4100 ) {
      ok( $dist == 0 );
      ok(
        join( ",", sort { $a cmp $b } @$name ) eq
          join( ",", sort { $a cmp $b } 'NR_FAKE3B', 'NR_FAKE3C' ) );

      # We de-dup, to unique values
      ok( join( ",", @$name2 ) eq 'FAKE3' );

      # if all values can be represented as scalars they are, else all represented
      # as arrays
      ok( $name2Gene eq 'FAKE3' );
    }

    # midopint to NR_FAKE4  chrM  + 4300  5000  3810  3900  1 1672, 3230, 0 FAKE4  NA  NA  NA  NA  NA  NA  NA  NA  NA  NA  NA
    if ( $pos < 3800 + ( 4300 - 3800 ) / 2 ) {

      #For single-point (from) nearest tracks, this case is equivalent to being
      #intergenic (or on top of txStart)
      ok( $distTss == 3800 - $pos );
      ok(
        join( ",", sort { $a cmp $b } @$nameTss ) eq
          join( ",", sort { $a cmp $b } 'NR_FAKE3', 'NR_FAKE3B', 'NR_FAKE3C' ) );
      ok( join( ",", @$name2Tss ) eq join( ",", @$name2 ) );
    }
    else {
      ok( $distTss == 4300 - $pos );
      ok( $nameTss eq 'NR_FAKE4' );
      ok( $name2Tss eq 'FAKE4' );
    }

    # end of NR_FAKE3B  chrM  + 3800  4100
    # to beginning of NR_FAKE3C  chrM  + 3800  4500
    if ( $pos >= 4100 ) {
      if ( $pos < 4300 ) {
        ok( $dist == 0 );
        ok( $name eq 'NR_FAKE3C' );
        ok( $name2 eq 'FAKE3' );

        # ok($distGene == 0);
        ok( $name2Gene eq $name2 );
      }

      # Within txEnd bound of NR_FAKE3C  chrM  + 3800  4500  3810  3900  1 1672, 3230, 0 FAKE3  NA  NA  NA  NA  NA  NA  NA  NA  NA  NA  NA
      # and txStart bound of NR_FAKE4  chrM  + 4300  5000  3810  3900  1 1672, 3230, 0 FAKE4  NA  NA  NA  NA  NA  NA  NA  NA  NA  NA  NA
      if ( $pos >= 4300 && $pos < 4500 ) {
        ok( $dist == 0 );
        ok(
          join( ",", sort { $a cmp $b } @$name ) eq
            join( ",", sort { $a cmp $b } 'NR_FAKE3C', 'NR_FAKE4' ) );
        ok(
          join( ",", sort { $a cmp $b } @$name2 ) eq
            join( ",", sort { $a cmp $b } 'FAKE3', 'FAKE4' ) );

        # ok($distGene == 0);
        ok(
          join( ",", sort { $a cmp $b } @$name2Gene ) eq
            join( ",", sort { $a cmp $b } @$name2 ) );
      }

      if ( $pos >= 4500 ) {
        ok( $dist == 0 );
        ok( $name eq 'NR_FAKE4' );
        ok( $name2 eq 'FAKE4' );

        # if(!defined $distGene) {
        #   p @outGene;
        # }
        # ok($distGene == 0);
        ok( $name2Gene eq $name2 );
      }
    }
  }

  # Intergenic, after the end
  if ( $pos >= 5000 ) {

    # 5000 is the end + 1 of the 0-based end
    ok( $dist == 4999 - $pos );
    ok( $name eq 'NR_FAKE4' );
    ok( $name2 eq 'FAKE4' );

    #For single-point (from) nearest tracks, this case is equivalent to being
    #intergenic (or on top of @2200) and past/on top of the last transcript (NR_FAKE2)
    ok( $distTss == 4300 - $pos );
    ok( $nameTss eq 'NR_FAKE4' );
    ok( $name2Tss eq 'FAKE4' );

    ok( !defined $name2Gene );

    # ok(!defined $distGene);
  }
}

done_testing();
