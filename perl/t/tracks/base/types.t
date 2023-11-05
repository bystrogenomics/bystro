use 5.10.0;
use strict;
use warnings;

use Test::More;

use Seq::Tracks::Base::Types;

my $rounder = Seq::Tracks::Base::Types->new();

my $rounded = $rounder->convert( "2.032", "number(2)" );

ok( $rounded == 2.0 );

$rounded = $rounder->convert( 2.032, "number(2)" );

ok( $rounded == 2.0 );

$rounded = $rounder->convert( 0.0000000023567, "number(3)" );

ok( $rounded == 0.00000000236 );

$rounded = $rounder->convert( 1.357e-10, "number(2)" );

ok( $rounded == 1.4e-10 );

done_testing();
