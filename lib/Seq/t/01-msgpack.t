use 5.10.0;
use strict;
use warnings;

use Test::More;
use DDP;

use Data::MessagePack;

my $mp = Data::MessagePack->new();
$mp->prefer_integer();

my $float =  sprintf("%0.6f", -3.563);
say "float is $float";

my $packed = $mp->pack($float );
ok(length($packed) == 11, "-3.563 as a 0.6f formatted string takes 11 bytes, 1 extra for sign");

$packed = $mp->pack( -3.563);
ok(length($packed) == 9, "floats are internally stored as doubles, take 9 bytes");


$packed = $mp->pack("1.000000");
ok(length($packed) == 10, "1.000000 float as a string takes 10 bytes");

$packed = $mp->pack("1.000000" + 0);
ok(length($packed) == 9, "1.000000 float as number (+0) takes 9 bytes");

$packed = $mp->pack(int("1.000000" + 0));
ok(length($packed) == 1, "1.000000 float as number truncated to int takes 1 byte");

done_testing();
# my $float2 =  "".3.56;

# my $packed2 = $mp->pack({0 => $float2});

# say "float 2 is, and takes " . length($packed2) . " bytes";
# p $packed2;

# my @array = ($float, "60.1");

# my $packed3 = $mp->pack(\@array);

# say "packed is";
# p $packed;
# say "length of packed is";
# my $length = length($packed);
# p $length;

# say "packed2 is";
# p $packed2;
# say "length of packed2 is";
# $length = length($packed2);
# p $length;

# say "packed3 is";
# p $packed3;

# # say "packed 3 is";
# # p $packed2;

# $float=  "". -0.24;

# $packed2 = $mp->pack($float);

# say "-0.24 stored as";
# p $packed2;

# say "length of -0.24 is : " . length($packed2) . " bytes";

# $float=  "".-2.24;

# $packed2 = $mp->pack($float);

# say "-2.243 stored as";
# p $packed2;

# say "length of -2.243 is : " . length($packed2) . " bytes";


# ##array vs perl pack

# my $array = 1500;

# my $packedByPerl = pack('S', $array);
# my $packedByMsg = $mp->pack($array);
# my $packedByMsgInArray = $mp->pack([ $array ]);
# my $aIna = [$array];
# p $aIna;
# my $asHash = {0 => 30, 1 => 1, 2 => 6};
# my $asHashWithArray = {0 => 30, 1 => 1, 2 => 6, 3 => $array};
# my $hashPackedByMsg = $mp->pack($asHash);
# my $hashPackedByMsgWithArray = $mp->pack($asHashWithArray);

# say "length of perl array $packedByPerl is " . length($packedByPerl);
# p $packedByPerl;
# say "length of msgpacked perl string " . length($mp->pack($packedByPerl));
# my $mps = $mp->pack($packedByPerl);
# p $mps;
# say "length of msgpack array $packedByMsg is" . length($packedByMsg);
# say "length of msgpack array in array $packedByMsgInArray is" . length($packedByMsgInArray);
# say "length of msgpack hash is" . length ($hashPackedByMsg);
# say "length of msgpack hash with array is" . length ($hashPackedByMsgWithArray);

# $array = [30,0];


# my $value = 255;
# $packed = $mp->pack($value);

# say "255 represented in " . length($packed) . ' bytes';

# $value = -32;
# $packed = $mp->pack($value);

# say "-32 represented in " . length($packed) . ' bytes';

# my $hash = {0 => 1, 1 => 2};
# $packed = $mp->pack($hash);

# say "2 member hash represented in " . length($packed). ' bytes';

# $array = [1,2];
# $packed = $mp->pack($array);

# say "2 member array represented in " . length($packed). ' bytes';
# my $unpacked = $mp->unpack($packed);
# p $unpacked;

# $hash = {0 => 1, 1 => 2, 2 => 3, 3 => 4, 4 => 5, 5 => 6, 6 => 7, 7 => 8};
# $packed = $mp->pack($hash);

# say "8 member hash represented in " . length($packed). ' bytes';

# $array = [1,2,3,4,5,6,7,8];
# $packed = $mp->pack($array);

# say "8 member array represented in " . length($packed). ' bytes';
# $unpacked = $mp->unpack($packed);
# p $unpacked;

# $hash = {0 => 1, 1 => 2, 2 => 3, 3 => 4, 4 => 5};
# $packed = $mp->pack($hash);

# say "5 member hash represented in " . length($packed). ' bytes';

# $array = [1,2,3,4,5];
# $packed = $mp->pack($array);

# say "5 member array represented in " . length($packed). ' bytes';
# $unpacked = $mp->unpack($packed);
# p $unpacked;

# $hash = {0 => 1, 1 => 2, 2 => 3, 4 => 5};
# $packed = $mp->pack($hash);

# say "4 member hash represented in " . length($packed). ' bytes';