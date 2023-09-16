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
        'someString',
        'someInt: int',
    ],
    local_files => ['fake'],
    files_dir => './t/tracks/build/db/raw',
});

# unit separator
my $overlap = "\x1f";

my $str = '1: Homo sapiens BRCA1/BRCA2-containing complex subunit 3 (BRCC3), transcript variant 2, mRNA. (from RefSeq NM_001018055)';
my $expected = '1: Homo sapiens BRCA1/BRCA2-containing complex subunit 3 (BRCC3), transcript variant 2, mRNA. (from RefSeq NM_001018055)';
my $res = $seq->coerceFeatureType('someString', $str);

#modifies passed string, and also returns the modified value
ok($res eq $str && $str eq $expected,'Can clean string containing /');

$str = '2: Homo sapiens BRCA1'. $overlap . 'BRCA2-containing complex subunit 3 (BRCC3), transcript variant 2, mRNA. (from RefSeq NM_001018055)';
$expected = '2: Homo sapiens BRCA1,BRCA2-containing complex subunit 3 (BRCC3), transcript variant 2, mRNA. (from RefSeq NM_001018055)';
$res = $seq->coerceFeatureType('someString', $str);

ok($res eq $str && $str eq $expected,'Can clean string containing \x1f');

$str = '3: Homo sapiens BRCA1'. $overlap . 'BRCA2-containing complex subunit 3 (BRCC3), transcript variant 2, mRNA. (from RefSeq NM_001018055)';
$expected = '3: Homo sapiens BRCA1,BRCA2-containing complex subunit 3 (BRCC3), transcript variant 2, mRNA. (from RefSeq NM_001018055)';
$res = $seq->coerceFeatureType('someString', $str);

ok($res eq $str && $str eq $expected, 'Can clean string containing 2 \x1f');

$str = '4: Homo sapiens BRCA1'. $overlap . '(BRCA2)-containing complex subunit 3 (BRCC3), transcript variant 2, mRNA. (from RefSeq NM_001018055)';
$expected = '4: Homo sapiens BRCA1,(BRCA2)-containing complex subunit 3 (BRCC3), transcript variant 2, mRNA. (from RefSeq NM_001018055)';
$res = $seq->coerceFeatureType('someString', $str);

ok($res eq $str && $str eq $expected,'Can clean string containing \x1f(');

$str = '5: Homo sapiens BRCA1'. $overlap . '.(BRCA2)-containing complex subunit 3 (BRCC3), transcript variant 2, mRNA. (from RefSeq NM_001018055)';
$expected = '5: Homo sapiens BRCA1,.(BRCA2)-containing complex subunit 3 (BRCC3), transcript variant 2, mRNA. (from RefSeq NM_001018055)';
$res = $seq->coerceFeatureType('someString', $str);

ok($res eq $str && $str eq $expected,'Can clean string containing \x1f.(');

$str = '6: Homo sapiens BRCA1|(BRCA2)-containing complex subunit 3 (BRCC3), transcript variant 2, mRNA. (from RefSeq NM_001018055)';
$expected = '6: Homo sapiens BRCA1,(BRCA2)-containing complex subunit 3 (BRCC3), transcript variant 2, mRNA. (from RefSeq NM_001018055)';
$res = $seq->coerceFeatureType('someString', $str);

ok($res eq $str && $str eq $expected,'Can clean string containing |(');

$str = '6: Homo sapiens BRCA1|BRCA2-containing complex subunit 3 (BRCC3), transcript variant 2, mRNA. (from RefSeq NM_001018055)';
$expected = '6: Homo sapiens BRCA1,BRCA2-containing complex subunit 3 (BRCC3), transcript variant 2, mRNA. (from RefSeq NM_001018055)';
$res = $seq->coerceFeatureType('someString', $str);

ok($res eq $str && $str eq $expected,'Can clean string containing |');

$str = '7: Homo sapiens BRCA1;BRCA2-containing complex subunit 3 (BRCC3), transcript variant 2, mRNA. (from RefSeq NM_001018055)';
$expected = '7: Homo sapiens BRCA1,BRCA2-containing complex subunit 3 (BRCC3), transcript variant 2, mRNA. (from RefSeq NM_001018055)';
$res = $seq->coerceFeatureType('someString', $str);

ok($res eq $str && $str eq $expected,'Can clean string containing ;');

$str = '8: Homo sapiens BRCA1;.BRCA2-containing complex subunit 3 (BRCC3), transcript variant 2, mRNA. (from RefSeq NM_001018055)';
$expected = '8: Homo sapiens BRCA1,.BRCA2-containing complex subunit 3 (BRCC3), transcript variant 2, mRNA. (from RefSeq NM_001018055)';
$res = $seq->coerceFeatureType('someString', $str);

ok($res eq $str && $str eq $expected, 'Can clean string containing ;.');

my $test='NA';
$res = $seq->coerceFeatureType('someString', $test);

ok(!defined $test && !defined $res);

$test='NA';
$res = $seq->coerceFeatureType('someString', $test);

ok(!defined $test && !defined $res);

$test='NA;';
$res = $seq->coerceFeatureType('someString', $test);

ok(!defined $test && !defined $res);

$test='NA|';
$res = $seq->coerceFeatureType('someString', $test);

ok(!defined $test && !defined $res);

$test='NA;';
$res = $seq->coerceFeatureType('someString', $test);

ok(!defined $test && !defined $res);

$test='NA' . $overlap;
$res = $seq->coerceFeatureType('someString', $test);

ok(!defined $test && !defined $res);

$test='NA' . $overlap;
$res = $seq->coerceFeatureType('someString', $test);

ok(!defined $test && !defined $res);

$test='';
$res = $seq->coerceFeatureType('someString', $test);

ok(!defined $test && !defined $res);

$test='.';
$res = $seq->coerceFeatureType('someString', $test);

ok(!defined $test && !defined $res);

my $delims = Seq::Output::Delimiters->new();
$test=$delims->emptyFieldChar;
$res = $seq->coerceFeatureType('someString', $test);

ok(!defined $test && !defined $res);

$expected = '. Hello World';
$test='. Hello World';
$res = $seq->coerceFeatureType('someString', $test);

ok($test eq $res && $res eq $expected, 'Doesn\'t strip valued sentences of undef-flagged characters');

done_testing();