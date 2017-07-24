#!/usr/bin/env perl
use 5.10.0;
use strict;
use warnings;
use Test::More;
use lib './lib';
use Mouse 2;
use Seq::Tracks::Base::Types;
use DDP;
use Scalar::Util qw/looks_like_number/;

my $converter = Seq::Tracks::Base::Types->new();

my $floatString = '.0000000000001';

my $floatConvert = $converter->convert($floatString, 'float');

my $numConvert = $converter->convert($floatString, 'number');

ok(looks_like_number($floatConvert), "float looks like number");
ok(looks_like_number($numConvert), "number looks like number");

ok($floatConvert == .0000000000001, "float doesn't round");
ok("$floatConvert" eq "1e-13", "perl represents large floats in scientific notation");
# Is too small to display by default in decimal notation
ok("$numConvert" eq '1e-13', "perl represents large floats in scientific notation");
ok("$numConvert" == .0000000000001, "number is a number");

my $exactFloat = '1.00000000';
$floatConvert = $converter->convert($exactFloat, 'float');
$numConvert = $converter->convert($exactFloat, 'number');

ok(looks_like_number($floatConvert), "float looks like number");
ok(looks_like_number($numConvert), "number looks like number");
ok("$floatConvert" == 1, "float converts ints");
ok("$numConvert" == 1, "number converts ints");

my $roundedFloat = '1.05';
$floatConvert = $converter->convert($roundedFloat, 'float');
$numConvert = $converter->convert($roundedFloat, 'number');

ok(looks_like_number($floatConvert), "float looks like number");
ok(looks_like_number($numConvert), "number looks like number");
ok("$floatConvert" eq "1.05", "float doesn't round, and keeps the minimum number of decimal places when conveted to string");
ok("$numConvert" eq "1.05", "number never rounds, and uses only necessary precision");


$roundedFloat = '1.0000005';
$floatConvert = $converter->convert($roundedFloat, 'float');
$numConvert = $converter->convert($roundedFloat, 'number');

ok(looks_like_number($floatConvert), "float looks like number");
ok(looks_like_number($numConvert), "number looks like number");
ok("$floatConvert" eq "1.0000005", "float doesn't round");
ok("$numConvert" eq "1.0000005", "number doesn't round");

$roundedFloat = '123000000.2590';
$floatConvert = $converter->convert($roundedFloat, 'float');
$numConvert = $converter->convert($roundedFloat, 'number');

ok(looks_like_number($floatConvert), "float looks like number");
ok(looks_like_number($numConvert), "number looks like number");
ok("$floatConvert" eq "123000000.259","Can convert number larger than 1 million to float");
ok("$numConvert" eq "123000000.259", "Can convert decimal containing number larger than 1 million");

$floatConvert = $converter->convert("1e-13", 'float');
$numConvert = $converter->convert("1e-13", 'number');
ok("$floatConvert" == .0000000000001, "float can convert scientific notation string");
ok($numConvert == .0000000000001, "number can convert scientific notation string");

done_testing();