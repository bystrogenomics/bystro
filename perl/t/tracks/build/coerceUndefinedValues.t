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

Seq::DBManager::initialize( { databaseDir => './t/tracks/build/db/index', } );

my $seq = Seq::Tracks::Build->new(
  {
    chromosomes => ['chr1'],
    name        => 'test',
    type        => 'gene',
    assembly    => 'hg38',
    features    => [ 'someString', 'someInt: int', ],
    local_files => ['fake'],
    files_dir   => './t/tracks/build/db/raw',
  }
);

my $delims = Seq::Output::Delimiters->new();

my $test = 'NA';
my $res  = $seq->_stripAndCoerceUndef($test);

ok( !defined $test && !defined $res, "Modifies passed value, and sets NA to undef" );

$test = '.';
$res  = $seq->_stripAndCoerceUndef($test);

ok( !defined $test && !defined $res, "Sets . to undef" );

$test = 'see cases';
$res  = $seq->_stripAndCoerceUndef($test);

ok( !defined $test && !defined $res, "'see cases' is not a valid value" );

$test = 'see cases ';
$res  = $seq->_stripAndCoerceUndef($test);

$test = 'unknown';
$res  = $seq->_stripAndCoerceUndef($test);

ok( !defined $test && !defined $res,
  "' unknown ' with leading/trailing whitespace not a valid value" );

$test = ' unknown ';
$res  = $seq->_stripAndCoerceUndef($test);

ok( !defined $test && !defined $res,
  "' unknown ' with leading/trailing whitespace not a valid value" );

$test = ' see cases';
$res  = $seq->_stripAndCoerceUndef($test);

ok( !defined $test && !defined $res,
  "'see cases' with leading whitespace not a valid value" );

$test = ' see cases ';
$res  = $seq->_stripAndCoerceUndef($test);

ok( !defined $test && !defined $res,
  "'see cases' with leading/trailing whitespace is not a valid value" );

$test = 'not provided';
$res  = $seq->_stripAndCoerceUndef($test);

ok( !defined $test && !defined $res, "'not provided' is not a valid value" );

$test = 'not specified';
$res  = $seq->_stripAndCoerceUndef($test);

ok( !defined $test && !defined $res, "'not specified' is not a valid value" );

$test = 'no assertion provided';
$res  = $seq->_stripAndCoerceUndef($test);

ok( !defined $test && !defined $res,
  "'no assertion provided' is not a valid value" );

$test = 'no assertion criteria provided';
$res  = $seq->_stripAndCoerceUndef($test);

ok( !defined $test && !defined $res,
  "'no assertion criteria provided' is not a valid value" );

$test = 'no interpretation for the single variant';
$res  = $seq->_stripAndCoerceUndef($test);

ok( !defined $test && !defined $res,
  "'no interpretation for the single variant' is not a valid value" );

$test = 'no assertion for the individual variant';
$res  = $seq->_stripAndCoerceUndef($test);

ok( !defined $test && !defined $res,
  "'no assertion for the individual variant' is not a valid value" );

$test = $delims->emptyFieldChar;
$res  = $seq->_stripAndCoerceUndef($test);

ok( !defined $test && !defined $res, "Sets the emptyFieldChar to undef" );

$test = ' NA ';
$res  = $seq->_stripAndCoerceUndef($test);

ok( !defined $test && !defined $res, "Whitespace doesnt affect coercion" );

my $expected = 'NA / Some value';
$test = 'NA / Some value';
$res  = $seq->_stripAndCoerceUndef($test);

ok( $test eq $res && $res eq $expected, "Doesn't clear valued statements" );

$test = " SOMETHING NOT NULL ";
$seq->_stripAndCoerceUndef($test);
ok( $test eq "SOMETHING NOT NULL",
  "_stripAndCoerceUndef also strips leading/trailing spaces" );

done_testing();
1;
