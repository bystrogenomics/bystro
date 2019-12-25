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

# my $baseMapper = Seq::Tracks::Reference::MapBases->new();
# my $siteTypes = Seq::Tracks::Gene::Site::SiteTypeMap->new();
Seq::DBManager::initialize({
    databaseDir => './t/tracks/build/db/index',
  });

my $seq = Seq::Tracks::Build->new({
    chromosomes => ['chr1'],
    name => 'test',
    type => 'gene',
    assembly => 'hg38',
    features => [
        'someFeature',
        'someOther',
        'someToSplit',
        'someToJoinLeft',
        'someToJoinRight'
    ],
    build_field_transformations => {
        someFeature => "replace /[.]+/,/",
        someOther => "replace /[.]+/ /",
        someToSplit => "split [,]+",
        someToJoinRight => ". _with_hello_world",
        someToJoinLeft => "chr ."
    },
    local_files => ['fake'],
    files_dir => './t/tracks/build/db/raw',
});

my $str = 'criteria_provided..multiple_submitters..no_conflicts';
my $cp = $str;
my $expected = 'criteria_provided,multiple_submitters,no_conflicts';
my $res = $seq->transformField('someFeature', $str);

ok($res eq $expected, "can replace multiple characters");
ok($str eq $cp, "doesn't modify input in place");

$str = 'criteria_provided..multiple_submitters..no_conflicts';
$expected = 'criteria_provided multiple_submitters no_conflicts';
$res = $seq->transformField('someOther', $str);

ok($res eq $expected, "can replace with spaces");

$str = 'something,to,,split';
my @exp= ('something', 'to', 'split');

$res = $seq->transformField('someToSplit', $str);

ok(
    @$res == 3
    && $res->[0] eq $exp[0]
    && $res->[1] eq $exp[1]
    && $res->[2] eq $exp[2],
    "can split on simple characters, with 1+ matches"
);


$str = 'some_long_string';
$expected = 'some_long_string_with_hello_world';
$res = $seq->transformField('someToJoinRight', $str);

ok($res eq $expected, "can join to right end");

$str = '1';
$expected = 'chr1';
$res = $seq->transformField('someToJoinLeft', $str);

ok($res eq $expected, "can join to left end");


done_testing();