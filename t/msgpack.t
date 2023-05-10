use 5.10.0;
use strict;
use warnings;

use Test::More;
use DDP;

use Data::MessagePack;

my $mp = Data::MessagePack->new()->prefer_integer()->prefer_float32();

my $float =  sprintf("%0.6f", -3.563);

my $packed = $mp->pack($float );
ok(length($packed) == 11, "-3.563 as a 0.6f formatted string takes 11 bytes, 1 extra for sign");

$packed = $mp->pack( -3.563);
ok(length($packed) == 5, "floats are internally stored 4 in 4bytes + 1 for schema");


$packed = $mp->pack("1.000000");
ok(length($packed) == 10, "1.000000 float as a string takes 10 bytes");

$packed = $mp->pack("1.000000" + 0);
ok(length($packed) == 5, "1.000000 float as number (+0) takes 5 bytes");

$packed = $mp->pack(int("1.000000" + 0));
ok(length($packed) == 1, "1.000000 float as number truncated to int takes 1 byte");

done_testing();