use 5.10.0;
use strict;
use warnings;
use Seq::Tracks::Score::Build::Round;
use Test::More;
use DDP;

plan tests => 4;

my $scalingFactor = 1000;
my $rounder =
  Seq::Tracks::Score::Build::Round->new( { scalingFactor => $scalingFactor } );

say "\n Testing rounder functionality \n";

ok(
  $rounder->round(.068) / $scalingFactor == .068,
  "as long as enough precision, no rounding"
);
ok(
  $rounder->round(.063) / $scalingFactor == .063,
  "as long as enough precision, no rounding"
);
ok(
  $rounder->round(.065) / $scalingFactor == .065,
  "as long as enough precision, no rounding"
);
ok(
  $rounder->round(-0.475554) / $scalingFactor == -0.476,
  "rounds beyond scaling factor precision to nearest digit"
);
