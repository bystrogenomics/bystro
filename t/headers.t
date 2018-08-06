use 5.10.0;
use strict;
use warnings;

use Test::More;
use Seq::Output;
use Seq::Headers;
use DDP;

my $head = Seq::Headers->new();

$head->addFeaturesToHeader(['first', 'second', 'third']);

my $str = $head->getString();
my $expected = "first\tsecond\tthird";
ok($str eq $expected, "Can write basic header");

my $arr = $head->getOrderedHeaderNoMap();
ok(join("\t", @$arr) eq $expected, "Store features in order given");

$head->addFeaturesToHeader(['really_first', 'really_second'], undef, 1);
$expected = "really_first\treally_second\t$expected";

$arr = $head->getOrderedHeaderNoMap();

ok(join("\t", @$arr) eq $expected, "Can prepend multiple features");

$head->addFeaturesToHeader('really_really_first', undef, 1);
$expected = "really_really_first\t$expected";

$arr = $head->getOrderedHeaderNoMap();
ok(join("\t", @$arr) eq $expected, "Can prepend features");

$head->initialize();
$arr = $head->getOrderedHeaderNoMap();
ok(@$arr == 0, "Can clear header");

$head->addFeaturesToHeader(['child1', 'child2', 'child3'], 'p1');
$expected = "child1\tchild2\tchild3";

$arr = $head->getOrderedHeaderNoMap();
p $arr;
ok(join("\t",@{$arr->[0]}) eq $expected, "Can create nested features");

my $idx = $head->getParentIndices();
p $idx;
ok($idx->{p1} == 0 && keys %$idx == 1, "Can recover top-level feature indices after addition of 1 feature");

$head->addFeaturesToHeader(['c1', 'c2', 'c3'], 'p2');
my $e2 = "c1\tc2\tc3";

$arr = $head->getOrderedHeaderNoMap();
p $arr;
ok(join("\t",@{$arr->[0]}) eq $expected && join("\t", @{$arr->[1]}) eq $e2, "Can add nested features");

$idx = $head->getParentIndices();
p $idx;
ok($idx->{p1} == 0 && $idx->{p2} == 1 && keys %$idx == 2, "Can recover top-level features after addition of 2nd feature");

$head->addFeaturesToHeader(['c1a', 'c2a', 'c3a'], 'p3', 1);
my $e3 = "c1a\tc2a\tc3a";

$arr = $head->getOrderedHeaderNoMap();
p $arr;
ok(join("\t", @{$arr->[0]}) eq $e3
&& join("\t",@{$arr->[1]}) eq $expected
&& join("\t", @{$arr->[2]}) eq $e2, "Can prepend nested features");

$idx = $head->getParentIndices();

ok($idx->{p3} == 0
&& $idx->{p1} == 1
&& $idx->{p2} == 2
&& keys %$idx == 3, "Can recover top-level features after addition of 3nd feature which is pre-pended");

my $p1 = $head->getParentFeatures('p1');
my $p2 = $head->getParentFeatures('p2');
my $p3 = $head->getParentFeatures('p3');

ok(join("\t", @$p1) eq $expected
&& join("\t", @$p2) eq $e2
&& join("\t", @$p3) eq $e3, "Can recover features by hash");

$str = $head->getString();

$expected = "p3.c1a\tp3.c2a\tp3.c3a\tp1.child1\tp1.child2\tp1.child3\tp2.c1\tp2.c2\tp2.c3";
ok($str eq $expected, "Can build string from nested features");

$head->addFeaturesToHeader('cadd');
$head->addFeaturesToHeader('phyloP', undef, 1);
$expected = "phyloP\t$expected\tcadd";

$str = $head->getString();

ok($str eq $expected, "Can mix and match nested features with non-nested");

$idx = $head->getParentIndices();

ok($idx->{phyloP} == 0
&& $idx->{p3} == 1
&& $idx->{p1} == 2
&& $idx->{p2} == 3
&& $idx->{cadd} == 4
&& keys %$idx == 5, "Can recover top-level feature indices after addition of non-nested features");

done_testing();
1;

