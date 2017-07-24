use 5.10.0;
use strict;
use warnings;

use Test::More;
use DDP;

use lib './lib';

use Data::MessagePack;
use Seq::Tracks::Base::Types;

my $mp = Data::MessagePack->new();
$mp->prefer_integer();

my $converter = Seq::Tracks::Base::Types->new();

my $numThatShouldBeInt  = '1.000000';

my $converted = $converter->convert($numThatShouldBeInt, 'number');

my $packed = $mp->pack($converted);

ok(length($packed) == 1, "The string $numThatShouldBeInt takes 1 byte in msgpack, when using the 'number' converter");

$converted = $converter->convert($numThatShouldBeInt, 'float');

$packed = $mp->pack($converted);

ok(length($packed) == 9, "The string $numThatShouldBeInt takes 9 bytes in msgpack, when using the 'float' converter");

$converted = $converter->convert($numThatShouldBeInt, 'int');

$packed = $mp->pack($converted);

ok(length($packed) == 1, "The string $numThatShouldBeInt takes 1 byte in msgpack, when using the 'int' converter");


$numThatShouldBeInt = "-1.000000";

$converted = $converter->convert($numThatShouldBeInt, 'number');

$packed = $mp->pack($converted);

ok(length($packed) == 1, "The string $numThatShouldBeInt takes 1 byte in msgpack, when using the 'number' converter");

$converted = $converter->convert($numThatShouldBeInt, 'int');

$packed = $mp->pack($converted);

ok(length($packed) == 1, "The string $numThatShouldBeInt takes 1 byte in msgpack, when using the 'number' converter");

$converted = $converter->convert($numThatShouldBeInt, 'float');
$packed = $mp->pack($converted);

ok(length($packed) == 1, "Perl does something bizarre. The string $numThatShouldBeInt takes 1 bytes in msgpack, when using the 'float' converter");

# Something is very wrong with Perl
$packed = $mp->pack($numThatShouldBeInt+ 0);
ok(length($packed) == 9, "Perl does something bizarre. The string $numThatShouldBeInt takes 9 bytes in msgpack, when using the 'float' converter operation, but not returning from the convert sub");


$packed = $mp->pack(-1);
ok(length($packed) == 1, "The number -1 takes 1 bytes in msgpack");


$packed = $mp->pack(1);
ok(length($packed) == 1, "The number 1 takes 9 bytes in msgpack");

$packed = $mp->pack("1.000000" + 0);
ok(length($packed) == 9, "The string '1.000000' + 0 takes 9 bytes in msgpack");

$packed = $mp->pack("-1.000000" + 0);
ok(length($packed) == 9, "The string -1.000000' + 0  takes 9 bytes in msgpack");


done_testing();

