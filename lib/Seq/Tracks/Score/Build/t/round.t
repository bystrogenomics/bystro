use 5.10.0;
use strict;
use warnings;

use Seq::Tracks::Score::Build::Round;
use Test::More;

plan tests => 7;

my $number = 1;
my $number2 = -1;
my $number3 = 1.000;
my $number4 = 1.253;
my $number5 = 10.253;
my $number6 = -10.253;
my $number7 = -1.253;

my $rounder = Seq::Tracks::Score::Build::Round->new();

ok($rounder->round($number) == 1);
ok($rounder->round($number2) == -1);
ok($rounder->round($number3) == 1);
ok($rounder->round($number4) == 1.25);
ok($rounder->round($number5) == 10.3);
ok($rounder->round($number6) == -10.3);
ok($rounder->round($number7) == -1.25);

use Data::MessagePack;
use DDP;

my $mp = Data::MessagePack->new();

$mp->prefer_integer();

my $rounded = $rounder->round(3.415);
my $packed = $mp->pack($rounded);

say "rounded length is " . length($packed);
p $rounded;
p $packed;
ok(length($packed) == 6);

my $arr = [$rounder->round(1.345), $rounder->round(2.32), $rounder->round(3.415)];
my $arr2 = [1.345, 2.32, 3.415];

$packed = $mp->pack($arr);
my $packed2 = $mp->pack($arr2);

ok(length($packed) < length($packed2), "rounding makes smaller");
say "length of not rounded array: ". length($packed2);
say "length of rounded array: " . length($packed);

say "difference in length is " . (length($packed2) - length($packed));
p $packed2;
p $packed;

$packed = $mp->pack(3.415);
say "length of packed float is: " . length($packed);
