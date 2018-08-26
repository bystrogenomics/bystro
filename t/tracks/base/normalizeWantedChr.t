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
my $wanted = $seq->normalizedWantedChr->{'chr1'};

ok($chr eq $wanted, "Won't modify listed chromosomes");


$wanted = $seq->normalizedWantedChr->{'1'};

ok($wanted eq '1', "Won't return prefix version if non-prefix version listed");

$wanted = $seq->normalizedWantedChr->{'chrM'};
ok($wanted eq 'chrM', "If MT given and chrM listed, returns chrM");

$wanted = $seq->normalizedWantedChr->{'MT'};
ok($wanted eq 'chrM', "If MT given and chrM listed, returns chrM");

$wanted = $seq->normalizedWantedChr->{'M'};
ok($wanted eq 'chrM', "If M given and chrM listed, returns chrM");

$wanted = $seq->normalizedWantedChr->{'chrMT'};
ok($wanted eq 'chrM', "If chrMT given and chrM listed, returns chrM");

$seq = Seq::Tracks::Base->new({
    chromosomes => [
        'chrMT',
        '1'
    ],
    name => 'test',
    type => 'gene',
    assembly => 'hg38'
});

$wanted = $seq->normalizedWantedChr->{'chrMT'};
ok($wanted eq 'chrMT', "If chrMT given and chrMT listed, returns chrMT");

$wanted = $seq->normalizedWantedChr->{'MT'};
ok($wanted eq 'chrMT', "If MT given and chrMT listed, returns chrMT");

$wanted = $seq->normalizedWantedChr->{'M'};
ok($wanted eq 'chrMT', "If M given and chrMT listed, returns chrMT");

$wanted = $seq->normalizedWantedChr->{'chrM'};
ok($wanted eq 'chrMT', "If chrM given and chrMT listed, returns chrMT");

$seq = Seq::Tracks::Base->new({
    chromosomes => [
        'MT',
        '1'
    ],
    name => 'test',
    type => 'gene',
    assembly => 'hg38'
});

$wanted = $seq->normalizedWantedChr->{'MT'};
ok($wanted eq 'MT', "If MT given and MT listed, returns MT");

$wanted = $seq->normalizedWantedChr->{'chrMT'};
ok($wanted eq 'MT', "If chrMT given and MT listed, returns MT");

$wanted = $seq->normalizedWantedChr->{'M'};
ok($wanted eq 'MT', "If M given and MT listed, returns MT");

$wanted = $seq->normalizedWantedChr->{'chrM'};
ok($wanted eq 'MT', "If chrM given and MT listed, returns MT");

$seq = Seq::Tracks::Base->new({
    chromosomes => [
        'M',
        '1'
    ],
    name => 'test',
    type => 'gene',
    assembly => 'hg38'
});

$wanted = $seq->normalizedWantedChr->{'MT'};
ok($wanted eq 'M', "If MT given and M listed, returns M");

$wanted = $seq->normalizedWantedChr->{'M'};
ok($wanted eq 'M', "If M given and M listed, returns M");

$wanted = $seq->normalizedWantedChr->{'chrMT'};
ok($wanted eq 'M', "If chrMT given and M listed, returns M");

$wanted = $seq->normalizedWantedChr->{'chrM'};
ok($wanted eq 'M', "If chrM given and M listed, returns M");


done_testing();