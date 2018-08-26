use 5.10.0;
use strict;
use warnings;

use Test::More;
use Seq::Tracks::Base;
use Seq::DBManager;
use Seq::Output::Delimiters;
use DDP;
# use Path::Tiny qw/path/;
# use Scalar::Util qw/looks_like_number/;
# use YAML::XS qw/LoadFile/;
# use DDP;
# use Seq::Tracks::Gene::Site::SiteTypeMap;
# use Seq::Tracks::Reference::MapBases;

# my $baseMapper = Seq::Tracks::Reference::MapBases->new();
# my $siteTypes = Seq::Tracks::Gene::Site::SiteTypeMap->new();
Seq::DBManager::initialize({
    databaseDir => './t/tracks/base/db/index',
});

my $seq = Seq::Tracks::Base->new({
    chromosomes => [
        'chr1',
        'chrM',
        '1'
    ],
    name => 'test',
    type => 'gene',
    assembly => 'hg38'
});

my $chr = 'chr1';
my $wanted = $seq->normalizedWantedChr->{$chr};

ok($chr eq $wanted, "Won't modify listed chromosomes");


$wanted = $seq->normalizedWantedChr->{'1'};

ok($wanted eq '1', "Won't return prefix version if non-prefix version listed");


$wanted = $seq->normalizedWantedChr->{'MT'};
ok($wanted eq 'chrM', "If MT given and chrM listed, returns chrM");

$wanted = $seq->normalizedWantedChr->{'M'};
ok($wanted eq 'chrM', "If M given and chrM listed, returns chrM");

done_testing();