use 5.10.0;
use strict;
use warnings;

use Test::More;
use Seq::Tracks::Build;
use Seq::DBManager;
use Seq::Output::Delimiters;
use DDP;
# use Path::Tiny qw/path/;
# use Scalar::Util qw/looks_like_number/;
# use YAML::XS qw/LoadFile/;
# use DDP;
# use Seq::Tracks::Gene::Site::SiteTypeMap;
# use Seq::Tracks::Reference::MapBases;

Seq::DBManager::initialize({
    databaseDir => './t/tracks/build/db/index',
  });

my $seq = Seq::Tracks::Build->new({
    chromosomes => ['chr1'],
    name => 'test',
    type => 'gene',
    assembly => 'hg38',
    features => [
        'someString',
        'someInt: int',
    ],
    local_files => ['fake'],
    files_dir => './t/tracks/build/db/raw',
});

my $delims = Seq::Output::Delimiters->new();

my $test='NA';
my $res = $seq->coerceUndefinedValues($test);

ok(!defined $test && !defined $res, "Modifies passed value, and sets NA to undef");

$test='.';
$res = $seq->coerceUndefinedValues($test);

ok(!defined $test && !defined $res, "Sets . to undef");


$test=$delims->emptyFieldChar;
$res = $seq->coerceUndefinedValues($test);

ok(!defined $test && !defined $res, "Sets the emptyFieldChar to undef");

$test=' NA ';
$res = $seq->coerceUndefinedValues($test);

ok(!defined $test && !defined $res, "Whitespace doesnt affect coercion");

my $expected = 'NA / Some value';
$test='NA / Some value';
$res = $seq->coerceUndefinedValues($test);

ok($test eq $res && $res eq $expected, "Doesn't clear valued statements");

done_testing();
1;