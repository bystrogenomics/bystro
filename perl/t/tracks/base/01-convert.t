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

my $floatString = '.0000000000001567';

my $floatConvert = $converter->convert( $floatString, 'number' );

my $roundedConvert = $converter->convert( $floatString, 'number(2)' );

ok( looks_like_number($floatConvert),   "number looks like number" );
ok( looks_like_number($roundedConvert), "number(2) looks like number" );

ok( $floatConvert == .0000000000001567, "number doesn't round" );
ok( "$floatConvert" eq "1.567e-13",
  "perl represents large floats in scientific notation" );

# Is too small to display by default in decimal notation
ok( $roundedConvert == .00000000000016, "number(2) rounds to 2 sigfigs" );

my $exactFloat = '1.00000000';
$floatConvert   = $converter->convert( $exactFloat, 'number' );
$roundedConvert = $converter->convert( $exactFloat, 'number(2)' );

ok( looks_like_number($floatConvert),   "number looks like number" );
ok( looks_like_number($roundedConvert), "number(2) looks like number" );
ok( $floatConvert == 1,
  "number converts floats to ints when exact solution possible" );
ok( $roundedConvert == 1,
  "number converts floats to ints when exact solution possible" );

my $roundedFloat = '1.05';
$floatConvert   = $converter->convert( $roundedFloat, 'number' );
$roundedConvert = $converter->convert( $roundedFloat, 'number(2)' );

ok( looks_like_number($floatConvert),   "number looks like number" );
ok( looks_like_number($roundedConvert), "number(2) looks like number" );
ok( $floatConvert == 1.05,              "number doesn't round" );
ok( $roundedConvert == 1.1,             "number(2) rounds to 2 sigfigs" );

$roundedFloat   = '1.0000005';
$floatConvert   = $converter->convert( $roundedFloat, 'number' );
$roundedConvert = $converter->convert( $roundedFloat, 'number(2)' );

ok( looks_like_number($floatConvert),   "float looks like number" );
ok( looks_like_number($roundedConvert), "number looks like number" );
ok( $floatConvert == 1.0000005,         "number doesn't round" );
ok( $roundedConvert == 1,               "number(2) rounds to 2 sigfigs" );

$roundedFloat   = '123000000.2590';
$floatConvert   = $converter->convert( $roundedFloat, 'number' );
$roundedConvert = $converter->convert( $roundedFloat, 'number(2)' );

ok( looks_like_number($floatConvert),   "float looks like number" );
ok( looks_like_number($roundedConvert), "number looks like number" );
ok( $floatConvert == 123000000.259,
  "number Can convert number larger than 1 million to float" );
ok( $roundedConvert == 120000000, "number(2) rounds to 2 sigfigs" );

$floatConvert   = $converter->convert( "1e-13", 'number' );
$roundedConvert = $converter->convert( "1e-13", 'number(2)' );
ok( $floatConvert == .0000000000001,
  "number can convert scientific notation string" );
ok(
  $roundedConvert == .0000000000001,
  "number(2) can convert scientific notation string; will not round if fewer sigfigs available than specified"
);

done_testing();
