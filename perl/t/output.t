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
  'withFeaturesTrack3' );
$head->addFeaturesToHeader('ref');

# trackOutIndices simply tracks Seq features apart from those passed in by
# a pre-processor
# this allows us to skip iterating over very long feature arrays on which we do no work
my $outputter =
  Seq::Output->new(
  { header => $head, trackOutIndices => [ 1, 2, 3, 4 ], refTrackName => 'ref' } );

my $delims = Seq::Output::Delimiters->new();
my $header = $head->getOrderedHeader();

ok( @$header == 5,          "Output header matches # of tracks" );
ok( @{ $header->[1] } == 3, "First package-defined track has 3 features" );
ok( !ref $header->[2],      "Second track has no features, is itself a feature" );
ok( @{ $header->[3] } == 3, "Third track has 2 features" );
ok( !ref $header->[4],      "Fourth track has no features, is itself a feature" );

my $hStr = $head->getString();

my @headFields = split( $delims->fieldSeparator, $hStr );

ok( @headFields == 9,
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
my @valsTrack3 = (
  'track3_feat1',
  'track3_feat2',
  [
    'track3_feat3_val1', [ 'track3_feat3_val2_overlap1', 'track3_feat3_val2_overlap2' ]
  ]
);

$row[3][0][$posIdx] = $valsTrack3[0];
$row[3][1][$posIdx] = $valsTrack3[1];
$row[3][2][$posIdx] = $valsTrack3[2];

# Add the reference
# Scalar tracks by definition have no features, and so in Bystro
# they are always 1 nested less deep
$row[4][0] = 'C';

my $nestedVals = join( $delims->overlapDelimiter, @{ $valsTrack3[2][1] } );
my $track3field3val =
  join( $delims->valueDelimiter, $valsTrack3[2][0], $nestedVals );

$expected .= $delims->fieldSeparator
  . join( $delims->fieldSeparator, @valsTrack3[ 0 .. 1 ], $track3field3val );

# Add ref
$expected .= $delims->fieldSeparator . "C" . "\n";

my @rows = ( \@row );

my $str = $outputter->makeOutputString( \@rows );

ok( $str eq $expected,
  "Can make complex output string with nested features, and with overlapping values" );

my @rowFields = split( $delims->fieldSeparator, $str );

ok( @headFields == @rowFields,
  "Output string length matches flattened header length" );

########## Test value deduplication in makeOutputString ##########
@row      = ( "somePreProcessorVal", [], [], [], [] );
$expected = "somePreProcessorVal" . $delims->fieldSeparator;

# Test all values duplicate
$row[1][0][0] = [ "t1_1a", "t1_1a" ];
# When both values are duplicate in an inner array, we expect a single value, with no overlap delimiter
$row[1][1][0] = [ [ "t1_2aa", "t1_2aa" ], [ "t1_2ba", "t1_2ba" ] ];
# If all values are duplicate across delimiters, we expect a single value, with no overlap delimiter or value delimiter
$row[1][2][0] =
  [ [ "t1_3aa", "t1_3aa" ], [ "t1_3aa", "t1_3aa", "t1_3aa", "t1_3aa" ] ];

# Track 1 values
$expected .= "t1_1a" . $delims->fieldSeparator; #$row[1][0][0]
$expected .=
    "t1_2aa"
  . $delims->valueDelimiter
  . "t1_2ba"
  . $delims->fieldSeparator;                    #$row[1][1][0]
$expected .= "t1_3aa" . $delims->fieldSeparator; #$row[1][1][0]

# We still handle scalar values just fine
$row[2][0] = [ "blah", "blah", "blah" ];

$expected .= "blah" . $delims->fieldSeparator;

# If not all values deuplicated, we won't deduplcate anything
$row[3][0][0] = [ "t3_1a", "t3_1a", "t3_1b" ];
# When not all values duplicated in inner array, we will not deduplicate
$row[3][1][0] = [ [ "t3_2aa", 't3_2aa', 't3_2ab' ], [ "t3_2ba", 't3_2ba' ] ];
# We will deduplicate values across position delimiters too
$row[3][2][0] = "t3_3a";
$row[3][2][1] = "t3_3a";

#$row[2][0][0]
$expected .= join( $delims->valueDelimiter, ( "t3_1a", "t3_1a", "t3_1b" ) )
  . $delims->fieldSeparator;
#$row[2][1][0]
$expected .=
    join( $delims->overlapDelimiter, ( "t3_2aa", 't3_2aa', 't3_2ab' ) )
  . $delims->valueDelimiter
  . "t3_2ba"
  . $delims->fieldSeparator;
#$row[2][2][0] & $row[2][2][1] are collapsed into a single value since they are the same
$expected .= "t3_3a" . $delims->fieldSeparator;

$row[4][0] = "T";

$expected .= "T" . "\n";

$str = $outputter->makeOutputString( [ \@row ] );

ok( $str eq $expected, "De-duplicates values" );

######### Test uniquefy  ##########
# Test 1: All identical defined values
my $result1 = Seq::Output::uniqueify( [ 'a', 'a', 'a' ] );
is_deeply( $result1, ['a'], "All identical values" );

# Test 2: All undefined values
my $result2 = Seq::Output::uniqueify( [ undef, undef, undef ] );
is_deeply( $result2, [undef], "All undefined values" );

# Test 3: Mix of undefined and defined values
my $result3 = Seq::Output::uniqueify( [ 'b', undef, 'b' ] );
is_deeply( $result3, [ 'b', undef, 'b' ], "Mix of undefined and defined values" );

# Test 4: Multiple distinct defined values
my $result4 = Seq::Output::uniqueify( [ 'c', 'd' ] );
is_deeply( $result4, [ 'c', 'd' ], "Multiple distinct values" );

# Test 5: Empty array
my $result5 = Seq::Output::uniqueify( [] );
is_deeply( $result5, [], "Empty array" );

done_testing();
1;
