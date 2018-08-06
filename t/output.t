use 5.10.0;
use strict;
use warnings;

use Test::More;
use Seq::Output;
use Seq::Headers;
use Seq::Output::Delimiters;

my $head = Seq::Headers->new();

$head->addFeaturesToHeader(['c1', 'c2', 'c3'], 'track1');
$head->addFeaturesToHeader('scalarTrack2');
$head->addFeaturesToHeader(['c1a', 'c2a'], 'track3');

my $outputter = Seq::Output->new({header => $head});

my $delims = Seq::Output::Delimiters->new();

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
my $expected;

my @t1_f1;
my @t1_f2;
my @t1_f3;

$t1_f1[0] = ['transcript1', 'transcript2'];
$expected = 'transcript1' . $delims->valueDelimiter . 'transcript2';

$t1_f2[0] = [ ['t1_val2_sub_sub1', 't1_val2_sub_sub2'], 't1_val2_sub1' ];
$expected .= $delims->fieldSeparator
          . "t1_val2_sub_sub1". $delims->overlapDelimiter . 't1_val2_sub_sub2'
          . $delims->valueDelimiter . 't1_val2_sub1';

$t1_f3[0] = ['t1_val3'];
$expected .= $delims->fieldSeparator
          . 't1_val3';

my @t1;
$t1[0] = \@t1_f1;
$t1[1] = \@t1_f2;
$t1[2] = \@t1_f3;

my @t2_f1;
$t2_f1[0] = 'scalar1';
$expected .= $delims->fieldSeparator
          . 'scalar1';

my @t2;
$t2[0] = \@t2_f1;

my @t3_f1;
$t3_f1[0] = 't3_val1';
$expected .= $delims->fieldSeparator
          . 't3_val1';

my @t3_f2;
$t3_f2[0] = ['t3_val2_sub1', 't3_val2_sub2'];
$expected .= $delims->fieldSeparator
          . 't3_val2_sub1' . $delims->valueDelimiter . 't3_val2_sub2'
          . "\n";

my @t3;
$t3[0] = \@t3_f1;
$t3[1] = \@t3_f2;

my @out;

#track1;
$out[0] = \@t1;
#track2;
$out[1] = \@t2;
#track3;
$out[2] = \@t3;


my $header = $head->getOrderedHeader();

ok(@$header == 3, "Output header matches # of tracks");
ok(@{$header->[0]} == 3, "First track has 3 features");
ok(!ref $header->[1], "Second track has 1 feature");
ok(@{$header->[2]} == 2, "Third track has 2 features");

my @rows;
$rows[0] = \@out;

my $str = $outputter->makeOutputString(\@rows);

ok($str eq $expected, "Can make complex output string with nested features, and with overlapping values");

my $hStr = $head->getString();

my @headFields = split($delims->fieldSeparator, $hStr);
my @rowFields = split($delims->fieldSeparator, $str);

ok(@headFields == @rowFields, "Output string length matches flattened header length");

done_testing();
1;

