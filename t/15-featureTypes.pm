#!/usr/bin/env/ perl
package MockDefinitions;
use 5.10.0;
use lib './lib';
use strict;
use warnings;
use Mouse;
with 'Seq::Tracks::Definition';

1;

package Test;

use 5.10.0;
use Test::More;
use strict;
use warnings;
use DDP;

plan tests => 4;

my $converter = MockDefinitions->new();

ok($converter->convert("200.39", "float") == 200.39, 'float conversion ok');

#we currently allow 6 decimal places
my $rounded = $converter->convert("200.3905056", "float");
ok($rounded == 200.390506, 'float rounding ok');
p $rounded;

$rounded = $converter->convert("200.3905055", "float");
ok($rounded == 200.390505, 'when 7th decimal value is 5, perl rounds down');
p $rounded;

ok($converter->convert("200.39", "int") == 200, 'int conversion ok');
