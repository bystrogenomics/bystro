use 5.10.0;
use strict;
use warnings;

use Test::More;

use Seq::Headers;
use Seq::Output::Delimiters;
use Seq::Output;

my $head = Seq::Headers->new();

$head->addFeaturesToHeader('preProcessorHeader1');
$head->addFeaturesToHeader( [ 'c1a', 'c1b', 'c1c' ], 'withFeaturesTrack1' );
$head->addFeaturesToHeader('scalarTrack2');
$head->addFeaturesToHeader( [ 'c2a', 'c2b', 'c2c_overlapped_vals' ],
  'withFeaturesTrack2' );

# trackOutIndices simply tracks Seq features apart from those passed in by
# a pre-processor
# this allows us to skip iterating over very long feature arrays on which we do no work
my $outputter =
  Seq::Output->new( { header => $head, trackOutIndices => [ 1, 2, 3 ] } );

my $delims = Seq::Output::Delimiters->new();

my $header = $head->getOrderedHeader();

ok( @$header == 4,          "Output header matches # of tracks" );
ok( @{ $header->[1] } == 3, "First package-defined track has 3 features" );
ok( !ref $header->[2],      "Second track has 1 feature" );
ok( @{ $header->[3] } == 3, "Third track has 2 features" );

my $hStr = $head->getString();

my @headFields = split( $delims->fieldSeparator, $hStr );

ok( @headFields == 8,
  "String header contains all expected fields, including those from pre-processor" );

# Everything is output as an array
# The first level is a track
# The 2nd level are the feature values

# Each feature value can have up to a depth of 3:
# 1) The position value [pos1, pos2, pos3] : only 1 pos for snp
# 2) The feature value at that position
# ###### Can have up to nesting of 2:
# ###### For instance, in refSeq, you may have:
# ###### transcript1;transcript2 transcript1_val1\\transcript1_val2;transcript2_onlyVal
# ###### Which is represented as
# ###### [ [transcript1, transcript2], [ [transcript1_val1, transcript1_val2], transcript2_onlyVal ] ]
# ###### Outer array is for feature
# ###### 1st inner array is for position (in indel)
# ###### 2nd inner array is for multiple values for that feature at that position
my $expected = "somePreProcessorVal";

my @t1_f1;
my @t1_f2;
my @t1_f3;

my @row = ( "somePreProcessorVal", [], [], [] );

# No indel
my $posIdx = 0;

my @valsTrack1 = ( 'transcript1', 'transcript2', "transcript3" );

$row[1][0][$posIdx] = $valsTrack1[0];
$row[1][1][$posIdx] = $valsTrack1[1];
$row[1][2][$posIdx] = $valsTrack1[2];

$expected .= $delims->fieldSeparator . join( $delims->fieldSeparator, @valsTrack1 );

my $valTrack2 = "someScalarVal1";

$row[2][$posIdx] = $valTrack2;

$expected .= $delims->fieldSeparator . $valTrack2;

# Separate delimiters so that one can clearly see that track3_feat3_*
# have a relationship with track3_feat3_val1
my @valsTrack2 = (
  'track3_feat1',
  'track3_feat2',
  [
    'track3_feat3_val1', [ 'track3_feat3_val2_overlap1', 'track3_feat3_val2_overlap2' ]
  ]
);

$row[3][0][$posIdx] = $valsTrack2[0];
$row[3][1][$posIdx] = $valsTrack2[1];
$row[3][2][$posIdx] = $valsTrack2[2];

my $nestedVals = join( $delims->overlapDelimiter, @{ $valsTrack2[2][1] } );
my $track3field3val =
  join( $delims->valueDelimiter, $valsTrack2[2][0], $nestedVals );

$expected .= $delims->fieldSeparator
  . join( $delims->fieldSeparator, @valsTrack2[ 0 .. 1 ], $track3field3val ) . "\n";

my @rows = ( \@row );

my $str = $outputter->makeOutputString( \@rows );

ok( $str eq $expected,
  "Can make complex output string with nested features, and with overlapping values" );

my @rowFields = split( $delims->fieldSeparator, $str );

ok( @headFields == @rowFields,
  "Output string length matches flattened header length" );

done_testing();
1;

