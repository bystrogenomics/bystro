use 5.10.0;
use strict;
use warnings;

use Test::More;

use Seq::Headers;
use Seq::Output;

my $head = Seq::Headers->new();

my %expected1Idx = (
  'first'  => 0,
  'second' => 1,
  'third'  => 2,
);

$head->addFeaturesToHeader( [ 'first', 'second', 'third' ] );

_checkFeatureIdx( $head, undef, \%expected1Idx, 'initial' );

my $str      = $head->getString();
my $expected = "first\tsecond\tthird";
ok( $str eq $expected, "Can write basic header" );

my $arr = $head->getOrderedHeader();
ok( join( "\t", @$arr ) eq $expected, "Store features in order given" );

$head->addFeaturesToHeader( [ 'really_first', 'really_second' ], undef, 1 );
$expected = "really_first\treally_second\t$expected";

%expected1Idx = (
  'really_first'  => 0,
  'really_second' => 1,
  'first'         => 2,
  'second'        => 3,
  'third'         => 4,
);

_checkFeatureIdx( $head, undef, \%expected1Idx,
  'prepended really_first, really_second' );

$arr = $head->getOrderedHeader();

ok( join( "\t", @$arr ) eq $expected, "Can prepend multiple features" );

$head->addFeaturesToHeader( 'really_really_first', undef, 1 );
$expected = "really_really_first\t$expected";

%expected1Idx = (
  'really_really_first' => 0,
  'really_first'        => 1,
  'really_second'       => 2,
  'first'               => 3,
  'second'              => 4,
  'third'               => 5,
);

_checkFeatureIdx( $head, undef, \%expected1Idx, 'prepended really_really_first' );

$arr = $head->getOrderedHeader();
ok( join( "\t", @$arr ) eq $expected, "Can prepend features" );

########### Cleared here so all earlier features go away ###############
$head->initialize();
$arr = $head->getOrderedHeader();
ok( @$arr == 0, "Can clear header" );

$head->addFeaturesToHeader( [ 'child1', 'child2', 'child3' ], 'p1' );
$expected = "child1\tchild2\tchild3";

# These are relative to their parent, or the root if no parent
%expected1Idx = (
  'p1' => { #0
    'child1' => 0,
    'child2' => 1,
    'child3' => 2,
  }
);

_checkFeatureIdx( $head, undef, \%expected1Idx, 'added p1' );

$arr = $head->getOrderedHeader();

ok( join( "\t", @{ $arr->[0] } ) eq $expected, "Can create nested features" );

my $idx = $head->getParentIndices();

ok( $idx->{p1} == 0 && keys %$idx == 1,
  "Can recover top-level feature indices after addition of 1 feature" );

$head->addFeaturesToHeader( [ 'c1', 'c2', 'c3' ], 'p2' );
my $e2 = "c1\tc2\tc3";

# These are relative to their parent, or the root if no parent
%expected1Idx = (
  'p1' => { #0
    'child1' => 0,
    'child2' => 1,
    'child3' => 2,
  },
  'p2' => { #1
    'c1' => 0,
    'c2' => 1,
    'c3' => 2,
  }
);

_checkFeatureIdx( $head, undef, \%expected1Idx, 'added p2' );

$arr = $head->getOrderedHeader();

ok( join( "\t", @{ $arr->[0] } ) eq $expected && join( "\t", @{ $arr->[1] } ) eq $e2,
  "Can add nested features" );

$idx = $head->getParentIndices();

ok(
  $idx->{p1} == 0 && $idx->{p2} == 1 && keys %$idx == 2,
  "Can recover top-level features after addition of 2nd feature"
);

# Prepend 3rd parent
$head->addFeaturesToHeader( [ 'c1a', 'c2a', 'c3a' ], 'p3', 1 );
my $e3 = "c1a\tc2a\tc3a";

# These are relative to their parent, or the root if no parent
%expected1Idx = (
  'p3' => { #0
    'c1a' => 0,
    'c2a' => 1,
    'c3a' => 2,
  },
  'p1' => { #1
    'child1' => 0,
    'child2' => 1,
    'child3' => 2,
  },
  'p2' => { #2
    'c1' => 0,
    'c2' => 1,
    'c3' => 2,
  },
);

_checkFeatureIdx( $head, undef, \%expected1Idx, 'prepended p3' );

$arr = $head->getOrderedHeader();

ok(
  join( "\t", @{ $arr->[0] } ) eq $e3
    && join( "\t", @{ $arr->[1] } ) eq $expected
    && join( "\t", @{ $arr->[2] } ) eq $e2,
  "Can prepend nested features"
);

my $fIdx  = $head->getFeatureIdx( 'p3', 'c1a' );
my $fIdx2 = $head->getFeatureIdx( 'p3', 'c2a' );
my $fIdx3 = $head->getFeatureIdx( 'p3', 'c3a' );

$idx = $head->getParentIndices();

ok(
  $idx->{p3} == 0 && $idx->{p1} == 1 && $idx->{p2} == 2 && keys %$idx == 3,
  "Can recover top-level features after addition of 3nd feature which is pre-pended"
);

my $p1 = $head->getParentFeatures('p1');
my $p2 = $head->getParentFeatures('p2');
my $p3 = $head->getParentFeatures('p3');

ok(
  join( "\t", @$p1 ) eq $expected
    && join( "\t", @$p2 ) eq $e2
    && join( "\t", @$p3 ) eq $e3,
  "Can recover features by hash"
);

$str = $head->getString();

$expected =
  "p3.c1a\tp3.c2a\tp3.c3a\tp1.child1\tp1.child2\tp1.child3\tp2.c1\tp2.c2\tp2.c3";
ok( $str eq $expected, "Can build string from nested features" );

$head->addFeaturesToHeader('cadd');
$head->addFeaturesToHeader( 'phyloP', undef, 1 );

$expected = "phyloP\t$expected\tcadd";

# These are relative to their parent, or the root if no parent
%expected1Idx = (
  'phyloP' => 0,
  'p3'     => { #1
    'c1a' => 0,
    'c2a' => 1,
    'c3a' => 2,
  },
  'p1' => {     #2
    'child1' => 0,
    'child2' => 1,
    'child3' => 2,
  },
  'p2' => {     #3
    'c1' => 0,
    'c2' => 1,
    'c3' => 2,
  },
  'cadd' => 4,
);

_checkFeatureIdx( $head, undef, \%expected1Idx, 'prepended phylP, added cadd' );

$str = $head->getString();

ok( $str eq $expected, "Can mix and match nested features with non-nested" );

$idx = $head->getParentIndices();

ok(
  $idx->{phyloP} == 0
    && $idx->{p3} == 1
    && $idx->{p1} == 2
    && $idx->{p2} == 3
    && $idx->{cadd} == 4
    && keys %$idx == 5,
  "Can recover top-level feature indices after addition of non-nested features"
);

sub _checkFeatureIdx {
  my ( $header, $parent, $expectedHref, $testName ) = @_;
  $testName //= 'test';

  for my $featureName ( keys %$expectedHref ) {
    my $eVal = $expectedHref->{$featureName};

    if ( ref $eVal ) {
      _checkFeatureIdx( $header, $featureName, $expectedHref->{$featureName}, $testName );
      next;
    }

    my $actual = $header->getFeatureIdx( $parent, $featureName );

    ok(
      defined $actual && $actual == $eVal,
      "Test: $testName. Can look up index of feature $featureName relative to parent "
        . ( defined $parent ? $parent : 'header root' )
        . " (index is $eVal)"
    );
  }
}

done_testing();
1;
