use 5.10.0;
use strict;
use warnings;

package MockBuilder;
use lib './lib';
use Mouse;
extends 'Seq::Base';

1;

use Test::More;
use Path::Tiny qw/path/;
use Scalar::Util qw/looks_like_number/;
use DDP;

my $baseMapper = Seq::Tracks::Reference::MapBases->new();

my $seq = MockBuilder->new_with_config({config => path('./t/tracks/nearest/test.yml')->absolute, debug => 1});

system('rm -rf ' . path($seq->database_dir)->child('*'));

my $tracks = $seq->tracksObj;
my $refBuilder = $tracks->getRefTrackBuilder();
my $nearestTssBuilder = $tracks->getTrackBuilderByName('refSeq.nearestTss');
my $nearestBuilder = $tracks->getTrackBuilderByName('refSeq.nearest');
my $geneBuilder = $tracks->getTrackBuilderByName('refSeq.gene');

my $nearestGetter = $tracks->getTrackGetterByName('refSeq.nearest');
my $nearestTssGetter = $tracks->getTrackGetterByName('refSeq.nearestTss');
my $geneGetter = $tracks->getTrackGetterByName('refSeq.gene');

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

my $mainDbAref = $db->dbReadAll('chrM');
my $regionDataAref = $db->dbReadAll('refSeq.nearest/chrM');

my $hasNearestCount = 0;
my $hasNearestTssCount = 0;
my $nearestDbName = $nearestBuilder->dbName;
my $nearestTssDbName = $nearestTssBuilder->dbName;
my $geneDbName = $geneBuilder->dbName;

for my $dbData (@$mainDbAref) {
  if(defined $dbData->[$nearestDbName]) {
    $hasNearestCount++;
  }

  if(defined $dbData->[$nearestTssDbName]) {
    $hasNearestTssCount++;
  }
}

ok($hasNearestCount == @$mainDbAref);
ok($hasNearestTssCount == @$mainDbAref);

for my $pos (0 .. $#$mainDbAref) {
  my $dbData = $mainDbAref->[$pos];

  my @out;
  my $nGeneData = $nearestGetter->get($dbData, 'chrM', 'C', 'A', 0, 0, \@out, $pos);

  my @outTss;
  my $nTssGeneData = $nearestTssGetter->get($dbData, 'chrM', 'C', 'A', 0, 0, \@outTss, $pos);

  my @outGene;
  my $geneData = $geneGetter->get($dbData, 'chrM', 'C', 'A', 0, 0, \@outGene, $pos);

  # config features order is name, name2, and then we add dist in the 3rd position
  # so 0 == name 1 == name2 2 == dist
  my $name = $out[0][0][0];
  my $name2 = $out[1][0][0];
  my $dist = $out[2][0][0];

  my $nameTss = $outTss[0][0][0];
  my $name2Tss = $outTss[1][0][0];
  my $distTss = $outTss[2][0][0];

  my $nameGene = $outGene[0][0][0];
  my $name2Gene = $outGene[1][0][0];
  my $distGene = $outGene[2][0][0];

  # intergenic
  if($pos < 1672) {
    ok(!defined $nameGene);
    ok(!defined $name2Gene);
    ok(!defined $distGene);
  }

   # An intergenic position
  # In all of these tests, we assume that the end position (here txEnd) is 0-based
  # This should be ensured by the nearest builder function
  if($pos <= 1672) {
    ok($dist == 1672 - $pos);
    ok($name eq 'NR_137295');
    ok($name2 eq 'RNR2');

    # for intergenic stuff, all should be identical between
    # nearest tracks that go from .. to and those with just one endpoint (from)
    ok($distTss == $dist);
    ok($nameTss eq $name);
    ok($name2Tss eq $name2);
  }

  # 2000 here is the txStart of NR_FAKE1, so -1 that is the last non-NR_FAKE1 base
  if($pos >= 1672 && $pos < 2000) {
    ok($dist == 0);
    ok($name eq 'NR_137295');
    ok($name2 eq 'RNR2');

    # same as nGene within the gene when storeNearest is false
    ok($nameGene eq $name);
    ok($name2Gene eq $name2);
    ok(!defined $distGene);

    # for spaces between two from points of adjacent transcripts/regions
    # nearest tracks of only 'from' coordinate should treat these as essentially
    # intergenic
    my $midPoint = 1672 + (2000 - 1672) / 2;
    if($pos < $midPoint) {
      ok($distTss == 1672 - $pos);
      ok($nameTss eq 'NR_137295');
      ok($name2Tss eq 'RNR2');
    } else {
      ok($distTss == 2000 - $pos);
      ok($nameTss eq 'NR_FAKE1');
      ok($name2Tss eq 'RNR2_FAKE1');
    }
  }

  # 2200 here is the txStart of NR_FAKE2
  if($pos >= 2000 && $pos < 2200) {
    ok($dist == 0);
    ok(join(",", @$name) eq 'NR_137295,NR_FAKE1');
    ok(join(",", @$name2) eq 'RNR2,RNR2_FAKE1');

    # same as nGene within the gene when storeNearest is false
    ok(join(",", @$nameGene) eq 'NR_137295,NR_FAKE1');
    ok(join(",", @$name2Gene) eq 'RNR2,RNR2_FAKE1');
    ok(!defined $distGene);

    # for spaces between two from points of adjacent transcripts/regions
    # nearest tracks of only 'from' coordinate should treat these as essentially
    # intergenic
    my $midPoint = 2000 + (2200 - 2000) / 2;
    if($pos < $midPoint) {
      ok($distTss == 2000 - $pos);
      ok($nameTss eq 'NR_FAKE1');
      ok($name2Tss eq 'RNR2_FAKE1');
    } else {
      ok($distTss == 2200 - $pos);
      ok($nameTss eq 'NR_FAKE2');
      ok($name2Tss eq 'RNR2_FAKE2');
    }
  }

  # 2300 is txEnd, so -1 for last pos in RNR2_FAKE1
  if($pos >= 2200 && $pos < 2300) {
    ok($dist == 0);
    ok(join(",", @$name) eq 'NR_137295,NR_FAKE1,NR_FAKE2');
    ok(join(",", @$name2) eq 'RNR2,RNR2_FAKE1,RNR2_FAKE2');

    # same as nGene within the gene when storeNearest is false
    ok(join(",", @$nameGene) eq 'NR_137295,NR_FAKE1,NR_FAKE2');
    ok(join(",", @$name2Gene) eq 'RNR2,RNR2_FAKE1,RNR2_FAKE2');
    ok(!defined $distGene);

    #For single-point (from) nearest tracks, this case is equivalent to being 
    #intergenic (or on top of @2200) and past/on top of the last transcript (NR_FAKE2)
    ok($distTss == 2200 - $pos);
    ok($nameTss eq 'NR_FAKE2');
    ok($name2Tss eq 'RNR2_FAKE2');
  }

  # 3230 is 0-based open interval (txEnd), so -1 that is the end of RNR2
  if($pos >= 2300 && $pos < 3230) {
    ok($dist == 0);
    ok(join(",", @$name) eq 'NR_137295,NR_FAKE2');
    ok(join(",", @$name2) eq 'RNR2,RNR2_FAKE2');

    # same as nGene within the gene when storeNearest is false
    ok(join(",", @$nameGene) eq 'NR_137295,NR_FAKE2');
    ok(join(",", @$name2Gene) eq 'RNR2,RNR2_FAKE2');
    ok(!defined $distGene);

    #For single-point (from) nearest tracks, this case is equivalent to being 
    #intergenic (or on top of @2200) and past/on top of the last transcript (NR_FAKE2)
    ok($distTss == 2200 - $pos);
    ok($nameTss eq 'NR_FAKE2');
    ok($name2Tss eq 'RNR2_FAKE2');
  }

  # Between 3230 txStart and 3400 - 1 txEnd (since the 3400 is 0-based, open)
  if($pos >= 3230 && $pos < 3400) {
    ok($dist == 0);
    ok($name eq 'NR_FAKE2');
    ok($name2 eq 'RNR2_FAKE2');

    ok($nameGene eq $name);
    ok($name2Gene eq $name2);
    ok(!defined $distGene);

    #For single-point (from) nearest tracks, this case is equivalent to being 
    #intergenic (or on top of @2200) and past/on top of the last transcript (NR_FAKE2)
    ok($distTss == 2200 - $pos);
    ok($nameTss eq 'NR_FAKE2');
    ok($name2Tss eq 'RNR2_FAKE2');
  }

  # Intergenic, afte the end
  if($pos >= 3400) {
    ok($dist == 3399 - $pos);
    ok($name eq 'NR_FAKE2');
    ok($name2 eq 'RNR2_FAKE2');

    #For single-point (from) nearest tracks, this case is equivalent to being 
    #intergenic (or on top of @2200) and past/on top of the last transcript (NR_FAKE2)
    ok($distTss == 2200 - $pos);
    ok($nameTss eq 'NR_FAKE2');
    ok($name2Tss eq 'RNR2_FAKE2');

    ok(!defined $nameGene);
    ok(!defined $name2Gene);
    ok(!defined $distGene);
  }
}

done_testing();