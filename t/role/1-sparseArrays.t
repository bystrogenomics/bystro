#!/usr/bin/perl
# your code goes here
use 5.10.0;
use Test::More;
use strict;
use warnings;

plan tests => 5;

my @arr = (0,1,2);
 
for (my $i = @arr; $i < 4; $i++) {
  push @arr, undef;
}
 
push @arr, 3;
 
ok($arr[0] == 0);
ok($arr[1] == 1);
ok($arr[2] == 2);
ok(!defined $arr[3]);
ok($arr[4] == 3);