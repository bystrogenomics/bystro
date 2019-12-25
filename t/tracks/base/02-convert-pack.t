use 5.10.0;
use strict;
use warnings;

use Test::More;
use DDP;

use lib './lib';

use Data::MessagePack;
use Seq::Tracks::Base::Types;

my $mp = Data::MessagePack->new();
$mp->prefer_integer()->prefer_float32();

my $converter = Seq::Tracks::Base::Types->new();

my $numThatShouldBeInt  = '1.000000';

my $converted = $converter->convert($numThatShouldBeInt, 'number');

my $packed = $mp->pack($converted);

ok(length($packed) == 1, "The string $numThatShouldBeInt takes 1 byte in msgpack, when using the 'number' converter");

$packed = $mp->pack($numThatShouldBeInt);
ok(length($packed) == 10, "mspgack will pack a string as a string");

$converted = $converter->convert('1.1', 'number');

$packed = $mp->pack($converted);

ok(length($packed) == 5, "With prefer_float32 floating point numbers will be packed in 5 bytes/single precision");

$packed = $mp->pack("1.000000" + 0);
ok(length($packed) == 5, "The string '1.000000' + 0 takes 5bytes in msgpack with prefer_float32");

$packed = $mp->pack("-1.000000" + 0);
ok(length($packed) == 5, "The string -1.000000' + 0  takes 5 bytes in msgpack with prefer_float32");


done_testing();

