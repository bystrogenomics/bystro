use 5.10.0;
use strict;
use warnings;

package TestMe;
use Test::More;
use lib './lib';
use Seq::Tracks::Build;
use DDP;

system('rm -rf ./t/tracks/db/index');

Seq::DBManager::setGlobalDatabaseDir('./t/tracks/db/index');

my $t = Seq::Tracks::Build->new({files_dir => './t/tracks/db/raw/', name => 'test', type => 'sparse', chromosomes => ['testChr'], assembly => 'hgTest'});

my($mergeFunc, $cleanUp) = $t->makeMergeFunc();
my @testVals = (67, 45, 22, 35);
my @testVals2 = (33, 25, 21, 65);

my $chr = 'testChr';
my $pos = 1;

my ($err, $result) = $mergeFunc->($chr, $pos, \@testVals, \@testVals2);

ok(join(',', @{$result->[0]}) eq join(',', 67, 33));
ok(join(',', @{$result->[1]}) eq join(',', 45, 25));
ok(join(',', @{$result->[2]}) eq join(',', 22, 21));
ok(join(',', @{$result->[3]}) eq join(',', 35, 65));

done_testing();