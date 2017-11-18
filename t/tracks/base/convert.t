package Test;
use lib './lib';

use 5.10.0;
use Test::More;
use strict;
use warnings;
use DDP;
use Seq::Tracks::Base::Types;

plan tests => 4;

my $converter = Seq::Tracks::Base::Types->new();

ok($converter->convert("200.39", "float") == 200.39, 'float conversion ok');

#we currently allow 6 decimal places
my $rounded = $converter->convert("200.3905056", "float");
ok($rounded == 200.3905056, 'float rounding ok');

$rounded = $converter->convert("200.3905055", "float");
ok($rounded == 200.3905055, 'when 7th decimal value is 5, perl rounds down');

ok($converter->convert("200.39", "int") == 200, 'int conversion ok');
