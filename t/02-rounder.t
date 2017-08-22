use 5.10.0;
use strict;
use warnings;
use lib './lib';
use Seq::Tracks::Score::Build::Round;
use Test::More;

plan tests => 4;

my $rounder = Seq::Tracks::Score::Build::Round->new();
say "\n Testing rounder functionality \n";
ok ($rounder->round(.068) + 0 == .07, "rounds up above midpoint");
ok ($rounder->round(.063) + 0 == .06, "rounds down below midpoint");
ok ($rounder->round(.065) + 0 == .07, "rounds up at midpoint");
ok ($rounder->round(-0.475354) + 0 == -0.48, "rounds negative numbers");