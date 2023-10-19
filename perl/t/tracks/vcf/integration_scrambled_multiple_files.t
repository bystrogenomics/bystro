use 5.10.0;
use strict;
use warnings;

package MockBuilder;

use Mouse;

extends 'Seq::Base';

1;

use Test::More;
use lib 't/lib';
use TestUtils qw/ HaveRequiredBinary PrepareConfigWithTempdirs /;

use Path::Tiny;
use Scalar::Util qw/looks_like_number/;

use Seq::Tracks::Reference::Build;
use Seq::Tracks::Reference::MapBases;
use Seq::Tracks::Vcf::Build;

# Check required binary is available
if ( !HaveRequiredBinary("bystro-vcf") ) {
  plan skip_all => "Testing relies on bystro-vcf binary, which is not present";
}

# create temp directories
my $dir = Path::Tiny->tempdir();

# prepare temp directory and make test config file
my $config_file = PrepareConfigWithTempdirs(
  't/tracks/vcf/test.scrambled_multiple_files.yml',
  't/tracks/vcf/raw', [ 'database_dir', 'files_dir', 'temp_dir' ],
  'files_dir',        $dir->stringify
);

my $baseMapper = Seq::Tracks::Reference::MapBases->new();

my $seq = MockBuilder->new_with_config( {config => $config_file} );

my $tracks     = $seq->tracksObj;
my $refBuilder = $tracks->getRefTrackBuilder();
my $refGetter  = $tracks->getRefTrackGetter();

$refBuilder->db->dbPatch(
  'chr22', $refBuilder->dbName,
  15927888 - 1,
  $baseMapper->baseMap->{'C'}
); #chr14:19792736-19792737 #same
$refBuilder->db->dbPatch(
  'chr22', $refBuilder->dbName,
  15927876 - 1,
  $baseMapper->baseMap->{'G'}
); #chr14:19792727 #same
$refBuilder->db->dbPatch(
  'chr22', $refBuilder->dbName,
  15927837 - 1,
  $baseMapper->baseMap->{'A'}
); #chr14:19792869-19792870 #same
$refBuilder->db->dbPatch(
  'chr22', $refBuilder->dbName,
  15927835 - 1,
  $baseMapper->baseMap->{'G'}
); #chr14:19792857-19792858 #same
$refBuilder->db->dbPatch(
  'chr22', $refBuilder->dbName,
  15927834 - 1,
  $baseMapper->baseMap->{'G'}
); #chr14:19792818-19792819 #same
$refBuilder->db->dbPatch(
  'chr22', $refBuilder->dbName,
  15927765 - 1,
  $baseMapper->baseMap->{'A'}
); #chr14:19792816-19792817 #same
$refBuilder->db->dbPatch(
  'chr22', $refBuilder->dbName,
  15927759 - 1,
  $baseMapper->baseMap->{'A'}
); #chr14:19792815-19792816 #same
$refBuilder->db->dbPatch(
  'chr22', $refBuilder->dbName,
  15927755 - 1,
  $baseMapper->baseMap->{'T'}
); #On #chr14:19792746-19792747 #same
$refBuilder->db->dbPatch(
  'chr22', $refBuilder->dbName,
  15927745 - 1,
  $baseMapper->baseMap->{'A'}
); #chr14:19792740-19792741 #same

$refBuilder->db->cleanUp();

my $dbVar = $refBuilder->db->dbReadOne( 'chr22', 15927888 - 1 );
ok( $refGetter->get($dbVar) eq 'C' );

$dbVar = $refBuilder->db->dbReadOne( 'chr22', 15927876 - 1 );
ok( $refGetter->get($dbVar) eq 'G' );

$dbVar = $refBuilder->db->dbReadOne( 'chr22', 15927837 - 1 );
ok( $refGetter->get($dbVar) eq 'A' );

$dbVar = $refBuilder->db->dbReadOne( 'chr22', 15927835 - 1 );
ok( $refGetter->get($dbVar) eq 'G' );

$dbVar = $refBuilder->db->dbReadOne( 'chr22', 15927834 - 1 );
ok( $refGetter->get($dbVar) eq 'G' );

$dbVar = $refBuilder->db->dbReadOne( 'chr22', 15927765 - 1 );
ok( $refGetter->get($dbVar) eq 'A' );

$dbVar = $refBuilder->db->dbReadOne( 'chr22', 15927759 - 1 );
ok( $refGetter->get($dbVar) eq 'A' );

$dbVar = $refBuilder->db->dbReadOne( 'chr22', 15927755 - 1 ); #on
ok( $refGetter->get($dbVar) eq 'T' );

$dbVar = $refBuilder->db->dbReadOne( 'chr22', 15927745 - 1 );
ok( $refGetter->get($dbVar) eq 'A' );

my $vcfBuilder = $tracks->getTrackBuilderByName('gnomad.genomes.scrambled');

$vcfBuilder->buildTrack();

my $vcf = $tracks->getTrackGetterByName('gnomad.genomes.scrambled');

my $db = Seq::DBManager->new();

############### Feature tests ################
# The vcf file contains the following items:
# Comma separated values are treated as belonging to diff alleles
# The G allele is the 2nd of the two at this site:
# C CACA,G
# Therefore, for this first test, we expect the 2nd of two values, when present
# AC=3,0;
# AF=1.81378e-04, 0.00000e+00;
# AN=16540;
# AC_AFR=0,0;
# AC_AMR=0,0;
# AC_ASJ=0,0;
# AC_EAS=0,0;
# AC_FIN=0,0;
# AC_NFE=3,0;
# AC_OTH=0,0;
# AC_Male=3,0;
# AC_Female=0,0;
# AN_AFR=3614;
# AN_AMR=528;
# AN_ASJ=180;
# AN_EAS=892;
# AN_FIN=2256;
# AN_NFE=8466;
# AN_OTH=604;
# AN_Male=9308;
# AN_Female=7232;
# AF_AFR=0.00000e+00, 0.00000e+00;
# AF_AMR=0.00000e+00, 0.00000e+00;
# AF_ASJ=0.00000e+00, 0.00000e+00;
# AF_EAS=0.00000e+00, 0.00000e+00;
# AF_FIN=0.00000e+00, 0.00000e+00;
# AF_NFE=3.54359e-04, 0.00000e+00;
# AF_OTH=0.00000e+00, 0.00000e+00;
# AF_Male=3.22303e-04, 0.00000e+00;
# AF_Female=0.00000e+00, 0.00000e+00;
# AS_FilterStatus=PASS,RF|AC0

# Notice that this site has an RF|AC0 value for AS_FilterStatus
# Therefore it doesn't pass

my $href = $db->dbReadOne( 'chr22', 15927888 - 1 );

#my ($vcf, $href, $chr, $refBase, $allele, $alleleIdx, $positionIdx, $outAccum) = @_;
# At this position we have CACA,G alleles
my $out = [];

# Vcf tracks are going to be treated differently from transcripts and sparse tracks
# They should return nothing for the nth position in a deletion
# Or if the allele doesn't match exactly.
$vcf->get( $href, 'chr22', 'C', 'G', 0, $out );

ok(
  @{$out} == 0,
  "Non PASS AS_FilterStatus causes alleles to be skipped in multiallelic (testing build_row_filters on INFO values)"
);

# my $numFeatures = scalar @{$vcf->features};
# ok(@{$out} == $numFeatures, "vcf array contians an entry for each requested feature");

# for my $feature (@{$vcf->features}) {
#   my $idx = $vcf->getFieldDbName($feature);

#   ok(@{$out->[$idx]} == 1, "Every feature is considered bi-allelelic (because alleles are decomposed into bi-allelic sites, with 1 entry");
#   ok(@{$out->[$idx][0]} == 1 && !ref $out->[$idx][0], "Every feature contains a single position's worth of data, and that value is scalar");
# }

# We define these as lower case in our test yaml, so that is how we will look them up
# No decision is made on behalf of the user how to name these; will be taken as defined

my $trTvIdx = $vcf->getFieldDbName('trTv');
my $idIdx   = $vcf->getFieldDbName('id');
my $acIdx   = $vcf->getFieldDbName('ac');
my $afIdx   = $vcf->getFieldDbName('af');
my $anIdx   = $vcf->getFieldDbName('an');

my $acAfrIdx    = $vcf->getFieldDbName('ac_afr');
my $acAmrIdx    = $vcf->getFieldDbName('ac_amr');
my $acAsjIdx    = $vcf->getFieldDbName('ac_asj');
my $acEasIdx    = $vcf->getFieldDbName('ac_eas');
my $acFinIdx    = $vcf->getFieldDbName('ac_fin');
my $acNfeIdx    = $vcf->getFieldDbName('ac_nfe');
my $acOthIdx    = $vcf->getFieldDbName('ac_oth');
my $acMaleIdx   = $vcf->getFieldDbName('ac_male');
my $acFemaleIdx = $vcf->getFieldDbName('ac_female');

my $anAfrIdx    = $vcf->getFieldDbName('an_afr');
my $anAmrIdx    = $vcf->getFieldDbName('an_amr');
my $anAsjIdx    = $vcf->getFieldDbName('an_asj');
my $anEasIdx    = $vcf->getFieldDbName('an_eas');
my $anFinIdx    = $vcf->getFieldDbName('an_fin');
my $anNfeIdx    = $vcf->getFieldDbName('an_nfe');
my $anOthIdx    = $vcf->getFieldDbName('an_oth');
my $anMaleIdx   = $vcf->getFieldDbName('an_male');
my $anFemaleIdx = $vcf->getFieldDbName('an_female');

my $afAfrIdx    = $vcf->getFieldDbName('af_afr');
my $afAmrIdx    = $vcf->getFieldDbName('af_amr');
my $afAsjIdx    = $vcf->getFieldDbName('af_asj');
my $afEasIdx    = $vcf->getFieldDbName('af_eas');
my $afFinIdx    = $vcf->getFieldDbName('af_fin');
my $afNfeIdx    = $vcf->getFieldDbName('af_nfe');
my $afOthIdx    = $vcf->getFieldDbName('af_oth');
my $afMaleIdx   = $vcf->getFieldDbName('af_male');
my $afFemaleIdx = $vcf->getFieldDbName('af_female');

# ok($out->[$trTvIdx][0] == 0, "indels and multiallelics have 0 trTv value");
# ok(!defined $out->[$idIdx][0], "correctly finds that this site has no rsID");
# ok($out->[$acIdx][0] == 0, "correctly finds first the G alleles ac value");
# ok($out->[$afIdx][0] == 0, "correctly finds first the G alleles af value");
# ok($out->[$anIdx][0] == 16540, "correctly finds first the G alleles an value, which has only one value across all alleles");

# ok($out->[$acAfrIdx][0] == 0, "correctly finds first the G alleles ac_afr value");
# ok($out->[$acAmrIdx][0] == 0, "correctly finds first the G alleles ac_amr value");
# ok($out->[$acAsjIdx][0] == 0, "correctly finds first the G alleles ac_asj value");
# ok($out->[$acEasIdx][0] == 0, "correctly finds first the G alleles ac_eas value");
# ok($out->[$acFinIdx][0] == 0, "correctly finds first the G alleles ac_fin value");
# ok($out->[$acNfeIdx][0] == 0, "correctly finds first the G alleles ac_nfe value");
# ok($out->[$acOthIdx][0] == 0, "correctly finds first the G alleles ac_oth value");
# ok($out->[$acMaleIdx][0] == 0, "correctly finds first the G alleles ac_male value");
# ok($out->[$acFemaleIdx][0] == 0, "correctly finds first the G alleles ac_female value");

# ok($out->[$anAfrIdx][0] == 3614, "correctly finds first the G alleles an_afr value, which has only one value across all alleles");
# ok($out->[$anAmrIdx][0] == 528, "correctly finds first the G alleles an_amr value, which has only one value across all alleles");
# ok($out->[$anAsjIdx][0] == 180, "correctly finds first the G alleles an_asj value, which has only one value across all alleles");
# ok($out->[$anEasIdx][0] == 892, "correctly finds first the G alleles an_eas value, which has only one value across all alleles");
# ok($out->[$anFinIdx][0] == 2256, "correctly finds first the G alleles an_fin value, which has only one value across all alleles");
# ok($out->[$anNfeIdx][0] == 8466, "correctly finds first the G alleles an_nfe value, which has only one value across all alleles");
# ok($out->[$anOthIdx][0] == 604, "correctly finds first the G alleles an_oth value, which has only one value across all alleles");
# ok($out->[$anMaleIdx][0] == 9308, "correctly finds first the G alleles an_male value, which has only one value across all alleles");
# ok($out->[$anFemaleIdx][0] == 7232, "correctly finds first the G alleles an_female value, which has only one value across all alleles");

# ok($out->[$afAfrIdx][0] == 0, "correctly finds first the G alleles af_afr value");
# ok($out->[$afAmrIdx][0] == 0, "correctly finds first the G alleles af_amr value");
# ok($out->[$afAsjIdx][0] == 0, "correctly finds first the G alleles af_asj value");
# ok($out->[$afEasIdx][0] == 0, "correctly finds first the G alleles af_eas value");
# ok($out->[$afFinIdx][0] == 0, "correctly finds first the G alleles af_fin value");
# ok($out->[$afNfeIdx][0] == 0, "correctly finds first the G alleles af_nfe value");
# ok($out->[$afOthIdx][0] == 0, "correctly finds first the G alleles af_oth value");
# ok($out->[$afMaleIdx][0] == 0, "correctly finds first the G alleles af_male value");
# ok($out->[$afFemaleIdx][0] == 0, "correctly finds first the G alleles af_female value");

############### Feature tests ################
# The vcf file contains the following items:
# Comma separated values are treated as belonging to diff alleles
# The +ACA allele is the 1st of the two at this site:
# C CACA,G
# Therefore, for this 2nd test, we expect the 1st of two values, when present
# AC=3,0;
# AF=1.81378e-04, 0.00000e+00;
# AN=16540;
# AC_AFR=0,0;
# AC_AMR=0,0;
# AC_ASJ=0,0;
# AC_EAS=0,0;
# AC_FIN=0,0;
# AC_NFE=3,0;
# AC_OTH=0,0;
# AC_Male=3,0;
# AC_Female=0,0;
# AN_AFR=3614;
# AN_AMR=528;
# AN_ASJ=180;
# AN_EAS=892;
# AN_FIN=2256;
# AN_NFE=8466;
# AN_OTH=604;
# AN_Male=9308;
# AN_Female=7232;
# AF_AFR=0.00000e+00, 0.00000e+00;
# AF_AMR=0.00000e+00, 0.00000e+00;
# AF_ASJ=0.00000e+00, 0.00000e+00;
# AF_EAS=0.00000e+00, 0.00000e+00;
# AF_FIN=0.00000e+00, 0.00000e+00;
# AF_NFE=3.54359e-04, 0.00000e+00;
# AF_OTH=0.00000e+00, 0.00000e+00;
# AF_Male=3.22303e-04, 0.00000e+00;
# AF_Female=0.00000e+00, 0.00000e+00;

# Vcf tracks are going to be treated differently from transcripts and sparse tracks
# They should return nothing for the nth position in a deletion
# Or if the allele doesn't match exactly.
$out = [];
$vcf->get( $href, 'chr22', 'C', '+ACA', 0, $out );

my $numFeatures = scalar @{ $vcf->features };
ok( @{$out} == $numFeatures,
  "vcf array contains an entry for each requested feature" );

for my $feature ( @{ $vcf->features } ) {
  my $idx = $vcf->getFieldDbName($feature);

  ok(
    @{ $out->[$idx] } == 1,
    "Every feature is considered bi-allelelic (because alleles are decomposed into bi-allelic sites, with 1 entry"
  );
  ok(
    !ref $out->[$idx][0],
    "Every feature contains a single position's worth of data, and that value is scalar"
  );
}

$out = [];
$vcf->get( $href, 'chr22', 'C', '+ACA', 1, $out );
ok(
  @$out == 0,
  "Vcf getter does not tile annotations by the position of the allele. Only accepts position index 0"
);

$out = [];
$vcf->get( $href, 'chr22', 'C', '+ACA', 0, $out );

# # Although we've deprecated single-line multiallelics, we must continue to support them
# # Until we've decided to drop support
# # Note, taht is for some reason a person wanted to fetch only one of the two alleles
# # the site would appear as a [feature1, featuer2] rather than [[feature1_allele1...], [feature1_allele2...]]
# for my $feature (@{$vcf->features}) {
#   my $idx = $vcf->getFieldDbName($feature);

#   ok(@{$out->[$idx]} == 2, "For multiallelic sites, where both alleles on a single input row, each feature is given an array of length == \# of alleles");
# }

ok( $out->[$trTvIdx][0] == 0,   "indels and multiallelics have 0 trTv value" );
ok( !defined $out->[$idIdx][0], "correctly finds that this site has no rsID" );
ok( $out->[$acIdx][0] == 3,     "correctly finds first the +ACA allele ac value" );
ok(
  $out->[$afIdx][0] == unpack( 'f', pack( 'f', 1.81378e-04 ) ),
  "correctly finds first the +ACA allele af value"
);
ok(
  $out->[$anIdx][0] == 16540,
  "correctly finds first the +ACA allele an value, which has only one value across all alleles"
);

ok( $out->[$acAfrIdx][0] == 0,
  "correctly finds first the +ACA allele ac_afr value" );
ok( $out->[$acAmrIdx][0] == 0,
  "correctly finds first the +ACA allele ac_amr value" );
ok( $out->[$acAsjIdx][0] == 0,
  "correctly finds first the +ACA allele ac_asj value" );
ok( $out->[$acEasIdx][0] == 0,
  "correctly finds first the +ACA allele ac_eas value" );
ok( $out->[$acFinIdx][0] == 0,
  "correctly finds first the +ACA allele ac_fin value" );
ok( $out->[$acNfeIdx][0] == 3,
  "correctly finds first the +ACA allele ac_nfe value" );
ok( $out->[$acOthIdx][0] == 0,
  "correctly finds first the +ACA allele ac_oth value" );
ok( $out->[$acMaleIdx][0] == 3,
  "correctly finds first the +ACA allele ac_male value" );
ok( $out->[$acFemaleIdx][0] == 0,
  "correctly finds first the +ACA allele ac_female value" );

ok(
  $out->[$anAfrIdx][0] == 3614,
  "correctly finds first the +ACA allele an_afr value, which has only one value across all alleles"
);
ok(
  $out->[$anAmrIdx][0] == 528,
  "correctly finds first the +ACA allele an_amr value, which has only one value across all alleles"
);
ok(
  $out->[$anAsjIdx][0] == 180,
  "correctly finds first the +ACA allele an_asj value, which has only one value across all alleles"
);
ok(
  $out->[$anEasIdx][0] == 892,
  "correctly finds first the +ACA allele an_eas value, which has only one value across all alleles"
);
ok(
  $out->[$anFinIdx][0] == 2256,
  "correctly finds first the +ACA allele an_fin value, which has only one value across all alleles"
);
ok(
  $out->[$anNfeIdx][0] == 8466,
  "correctly finds first the +ACA allele an_nfe value, which has only one value across all alleles"
);
ok(
  $out->[$anOthIdx][0] == 604,
  "correctly finds first the +ACA allele an_oth value, which has only one value across all alleles"
);
ok(
  $out->[$anMaleIdx][0] == 9308,
  "correctly finds first the +ACA allele an_male value, which has only one value across all alleles"
);
ok(
  $out->[$anFemaleIdx][0] == 7232,
  "correctly finds first the +ACA allele an_female value, which has only one value across all alleles"
);

ok( $out->[$afAfrIdx][0] == 0,
  "correctly finds first the +ACA allele af_afr value" );
ok( $out->[$afAmrIdx][0] == 0,
  "correctly finds first the +ACA allele af_amr value" );
ok( $out->[$afAsjIdx][0] == 0,
  "correctly finds first the +ACA allele af_asj value" );
ok( $out->[$afEasIdx][0] == 0,
  "correctly finds first the +ACA allele af_eas value" );
ok( $out->[$afFinIdx][0] == 0,
  "correctly finds first the +ACA allele af_fin value" );
ok(
  $out->[$afNfeIdx][0] == unpack( 'f', pack( 'f', 3.54359e-04 ) ),
  "correctly finds first the +ACA allele af_nfe value"
);
ok( $out->[$afOthIdx][0] == 0,
  "correctly finds first the +ACA allele af_oth value" );
ok(
  $out->[$afMaleIdx][0] == unpack( 'f', pack( 'f', 3.22303e-04 ) ),
  "correctly finds first the +ACA allele af_male value"
);
ok( $out->[$afFemaleIdx][0] == 0,
  "correctly finds first the +ACA allele af_female value" );

##### TESTING VARIANTS THAT AREN'T PASS OR .
# chr22 15927755  . T G 296.53  NON_PASS
$out  = [];
$href = $db->dbReadOne( 'chr22', 15927755 - 1 );
$vcf->get( $href, 'chr22', 'T', 'G', 0, $out );

ok( @$out == 0, 'NON PASS/. variants are skipped' );

# Next let's check variants that are bi-allelic
# chr22 15927745  . A C 718.20  PASS
# AC=2;
# AF=6.93049e-05;
# AN=28858;
# AC_AFR=0;
# AC_AMR=0;
# AC_ASJ=0;
# AC_EAS=0;
# AC_FIN=0;
# AC_NFE=2;
# AC_OTH=0;
# AC_Male=1;
# AC_Female=1;
# AN_AFR=8454;
# AN_AMR=782;
# AN_ASJ=232;
# AN_EAS=1606;
# AN_FIN=3132;
# AN_NFE=13774;
# AN_OTH=878;
# AN_Male=15900;
# AN_Female=12958;
# AF_AFR=0.00000e+00;
# AF_AMR=0.00000e+00;
# AF_ASJ=0.00000e+00;
# AF_EAS=0.00000e+00;
# AF_FIN=0.00000e+00;
# AF_NFE=1.45201e-04;
# AF_OTH=0.00000e+00;
# AF_Male=6.28931e-05;
# AF_Female=7.71724e-05;

$out  = [];
$href = $db->dbReadOne( 'chr22', 15927745 - 1 );
$vcf->get( $href, 'chr22', 'A', 'C', 0, $out );

ok( $out->[$trTvIdx][0] == 2,   "A->C is a transversion, so given value of 2" );
ok( !defined $out->[$idIdx][0], "correctly finds that this site has no rsID" );
ok( $out->[$acIdx][0] == 2, "correctly finds the ac value for a biallelic site" );
ok(
  $out->[$afIdx][0] == unpack( 'f', pack( 'f', 6.93049e-05 ) ),
  "correctly finds the af value for a biallelic site"
);
ok( $out->[$anIdx][0] == 28858,
  "correctly finds the an value for a biallelic site" );

ok( $out->[$acAfrIdx][0] == 0,  "correctly finds the C allele ac_afr value" );
ok( $out->[$acAmrIdx][0] == 0,  "correctly finds C allele allele ac_amr value" );
ok( $out->[$acAsjIdx][0] == 0,  "correctly finds C allele allele ac_asj value" );
ok( $out->[$acEasIdx][0] == 0,  "correctly finds C allele allele ac_eas value" );
ok( $out->[$acFinIdx][0] == 0,  "correctly finds C allele allele ac_fin value" );
ok( $out->[$acNfeIdx][0] == 2,  "correctly finds C allele allele ac_nfe value" );
ok( $out->[$acOthIdx][0] == 0,  "correctly finds C allele allele ac_oth value" );
ok( $out->[$acMaleIdx][0] == 1, "correctly finds C allele allele ac_male value" );
ok( $out->[$acFemaleIdx][0] == 1,
  "correctly finds C allele allele ac_female value" );

ok(
  $out->[$anAfrIdx][0] == 8454,
  "correctly finds the C allele an_afr value, which has only one value across all alleles"
);
ok(
  $out->[$anAmrIdx][0] == 782,
  "correctly finds the C allele an_amr value, which has only one value across all alleles"
);
ok(
  $out->[$anAsjIdx][0] == 232,
  "correctly finds the C allele an_asj value, which has only one value across all alleles"
);
ok(
  $out->[$anEasIdx][0] == 1606,
  "correctly finds the C allele an_eas value, which has only one value across all alleles"
);
ok(
  $out->[$anFinIdx][0] == 3132,
  "correctly finds the C allele an_fin value, which has only one value across all alleles"
);
ok(
  $out->[$anNfeIdx][0] == 13774,
  "correctly finds the C allele an_nfe value, which has only one value across all alleles"
);
ok(
  $out->[$anOthIdx][0] == 878,
  "correctly finds the C allele an_oth value, which has only one value across all alleles"
);
ok(
  $out->[$anMaleIdx][0] == 15900,
  "correctly finds the C allele an_male value, which has only one value across all alleles"
);
ok(
  $out->[$anFemaleIdx][0] == 12958,
  "correctly finds the C allele an_female value, which has only one value across all alleles"
);

ok( $out->[$afAfrIdx][0] == 0, "correctly finds the C allele af_afr value" );
ok( $out->[$afAmrIdx][0] == 0, "correctly finds the C allele af_amr value" );
ok( $out->[$afAsjIdx][0] == 0, "correctly finds the C allele af_asj value" );
ok( $out->[$afEasIdx][0] == 0, "correctly finds the C allele af_eas value" );
ok( $out->[$afFinIdx][0] == 0, "correctly finds the C allele af_fin value" );
ok( $out->[$afNfeIdx][0] == unpack( 'f', pack( 'f', 1.45201e-04 ) ),
  "correctly finds the C allele af_nfe value" );
ok( $out->[$afOthIdx][0] == 0, "correctly finds the C allele af_oth value" );
ok( $out->[$afMaleIdx][0] == unpack( 'f', pack( 'f', 6.28931e-05 ) ),
  "correctly finds the C allele af_male value" );
ok( $out->[$afFemaleIdx][0] == unpack( 'f', pack( 'f', 7.71724e-05 ) ),
  "correctly finds the C allele af_female value" );

$out = [];

$href = $db->dbReadOne( 'chr22', 15927745 - 1 );
$vcf->get( $href, 'chr22', 'A', 'T', 0, $out );

ok( @$out == 0, "Alelleles that don't match are skipped" );

# Next let's check a variant that has 2 non-reference alleles, but which came from different lines
# The order of the lines doesn't matter
# TODO: write explicit test for order not mattering
# The first one

# chr22 15927834  . G T 183.64  PASS
# AC=1;
# AF=4.21905e-05;
# AN=23702;
# AC_AFR=1;
# AC_AMR=0;
# AC_ASJ=0;
# AC_EAS=0;
# AC_FIN=0;
# AC_NFE=0;
# AC_OTH=0;
# AC_Male=1;
# AC_Female=0;
# AN_AFR=6452;
# AN_AMR=638;
# AN_ASJ=222;
# AN_EAS=1398;
# AN_FIN=2270;
# AN_NFE=12010;
# AN_OTH=712;
# AN_Male=13204;
# AN_Female=10498;
# AF_AFR=1.54991e-04;
# AF_AMR=0.00000e+00;
# AF_ASJ=0.00000e+00;
# AF_EAS=0.00000e+00;
# AF_FIN=0.00000e+00;
# AF_NFE=0.00000e+00;
# AF_OTH=0.00000e+00;
# AF_Male=7.57346e-05;
# AF_Female=0.00000e+00;

$out = [];
my $firstAllele = [];

$href = $db->dbReadOne( 'chr22', 15927834 - 1 );
$vcf->get( $href, 'chr22', 'G', 'T', 0, $firstAllele );

ok(
  $firstAllele->[$trTvIdx][0] == 2, "trTv is 0 for multiallelics.
  However, when our vcf parser is passed 2 alleles at the same position, but on different source file lines, it treats that as a SNP.
  Therefore it calls it as a G->T transversion, or 2"
);
ok( !defined $firstAllele->[$idIdx][0],
  "correctly finds that this site has no rsID" );
ok( $firstAllele->[$acIdx][0] == 1,
  "correctly finds the ac value for a biallelic site" );
ok( $firstAllele->[$afIdx][0] == unpack( 'f', pack( 'f', 4.21905e-05 ) ),
  "correctly finds the af value for a biallelic site" );
ok( $firstAllele->[$anIdx][0] == 23702,
  "correctly finds the an value for a biallelic site" );

ok( $firstAllele->[$acAfrIdx][0] == 1, "correctly finds the T allele ac_afr value" );
ok( $firstAllele->[$acAmrIdx][0] == 0,
  "correctly finds T allele allele ac_amr value" );
ok( $firstAllele->[$acAsjIdx][0] == 0,
  "correctly finds T allele allele ac_asj value" );
ok( $firstAllele->[$acEasIdx][0] == 0,
  "correctly finds T allele allele ac_eas value" );
ok( $firstAllele->[$acFinIdx][0] == 0,
  "correctly finds T allele allele ac_fin value" );
ok( $firstAllele->[$acNfeIdx][0] == 0,
  "correctly finds T allele allele ac_nfe value" );
ok( $firstAllele->[$acOthIdx][0] == 0,
  "correctly finds T allele allele ac_oth value" );
ok( $firstAllele->[$acMaleIdx][0] == 1,
  "correctly finds T allele allele ac_male value" );
ok( $firstAllele->[$acFemaleIdx][0] == 0,
  "correctly finds T allele allele ac_female value" );

ok(
  $firstAllele->[$anAfrIdx][0] == 6452,
  "correctly finds the T allele an_afr value, which has only one value across all alleles"
);
ok(
  $firstAllele->[$anAmrIdx][0] == 638,
  "correctly finds the T allele an_amr value, which has only one value across all alleles"
);
ok(
  $firstAllele->[$anAsjIdx][0] == 222,
  "correctly finds the T allele an_asj value, which has only one value across all alleles"
);
ok(
  $firstAllele->[$anEasIdx][0] == 1398,
  "correctly finds the T allele an_eas value, which has only one value across all alleles"
);
ok(
  $firstAllele->[$anFinIdx][0] == 2270,
  "correctly finds the T allele an_fin value, which has only one value across all alleles"
);
ok(
  $firstAllele->[$anNfeIdx][0] == 12010,
  "correctly finds the T allele an_nfe value, which has only one value across all alleles"
);
ok(
  $firstAllele->[$anOthIdx][0] == 712,
  "correctly finds the T allele an_oth value, which has only one value across all alleles"
);
ok(
  $firstAllele->[$anMaleIdx][0] == 13204,
  "correctly finds the T allele an_male value, which has only one value across all alleles"
);
ok(
  $firstAllele->[$anFemaleIdx][0] == 10498,
  "correctly finds the T allele an_female value, which has only one value across all alleles"
);

ok( $firstAllele->[$afAfrIdx][0] == unpack( 'f', pack( 'f', 1.54991e-04 ) ),
  "correctly finds the T allele af_afr value" );
ok( $firstAllele->[$afAmrIdx][0] == 0, "correctly finds the T allele af_amr value" );
ok( $firstAllele->[$afAsjIdx][0] == 0, "correctly finds the T allele af_asj value" );
ok( $firstAllele->[$afEasIdx][0] == 0, "correctly finds the T allele af_eas value" );
ok( $firstAllele->[$afFinIdx][0] == 0, "correctly finds the T allele af_fin value" );
ok( $firstAllele->[$afNfeIdx][0] == 0, "correctly finds the T allele af_nfe value" );
ok( $firstAllele->[$afAsjIdx][0] == 0, "correctly finds the T allele af_asj value" );
ok( $firstAllele->[$afOthIdx][0] == 0, "correctly finds the T allele af_oth value" );
ok( $firstAllele->[$afMaleIdx][0] == unpack( 'f', pack( 'f', 7.57346e-05 ) ),
  "correctly finds the T allele af_male value" );
ok(
  $firstAllele->[$afFemaleIdx][0] == 0,
  "correctly finds the T allele af_female value"
);

# The 2nd one:
# chr22 15927834  rs199856444 G C 1458410.68  PASS
# AC=5232;
# AF=2.00721e-01;
# AN=26066;
# AC_AFR=1195;
# AC_AMR=199;
# AC_ASJ=48;
# AC_EAS=462;
# AC_FIN=539;
# AC_NFE=2634;
# AC_OTH=155;
# AC_Male=2860;
# AC_Female=2372;
# AN_AFR=7838;
# AN_AMR=630;
# AN_ASJ=216;
# AN_EAS=1372;
# AN_FIN=2596;
# AN_NFE=12638;
# AN_OTH=776;
# AN_Male=14358;
# AN_Female=11708;
# AF_AFR=1.52462e-01;
# AF_AMR=3.15873e-01;
# AF_ASJ=2.22222e-01;
# AF_EAS=3.36735e-01;
# AF_FIN=2.07627e-01;
# AF_NFE=2.08419e-01;
# AF_OTH=1.99742e-01;
# AF_Male=1.99192e-01;
# AF_Female=2.02597e-01;

my $secondAllele = [];
$href = $db->dbReadOne( 'chr22', 15927834 - 1 );
$vcf->get( $href, 'chr22', 'G', 'C', 0, $secondAllele );

ok(
  $secondAllele->[$trTvIdx][0] == 2, "trTv is 0 for multiallelics. \
  However, when our vcf parser is passed 2 alleles at the same position, but on different source file lines, it treats that as a SNP.
  Therefore it calls it as a G->C transversion, or 2"
);
ok(
  $secondAllele->[$idIdx][0] eq 'rs199856444',
  "correctly finds that this site has an rsID"
);
ok( $secondAllele->[$acIdx][0] == 5232,
  "correctly finds the ac value for a biallelic site" );
ok( $secondAllele->[$afIdx][0] == unpack( 'f', pack( 'f', 2.00721e-01 ) ),
  "correctly finds the af value for a biallelic site" );
ok( $secondAllele->[$anIdx][0] == 26066,
  "correctly finds the an value for a biallelic site" );

ok(
  $secondAllele->[$acAfrIdx][0] == 1195,
  "correctly finds the C allele ac_afr value"
);
ok(
  $secondAllele->[$acAmrIdx][0] == 199,
  "correctly finds C allele allele ac_amr value"
);
ok( $secondAllele->[$acAsjIdx][0] == 48,
  "correctly finds C allele allele ac_asj value" );
ok(
  $secondAllele->[$acEasIdx][0] == 462,
  "correctly finds C allele allele ac_eas value"
);
ok(
  $secondAllele->[$acFinIdx][0] == 539,
  "correctly finds C allele allele ac_fin value"
);
ok(
  $secondAllele->[$acNfeIdx][0] == 2634,
  "correctly finds C allele allele ac_nfe value"
);
ok(
  $secondAllele->[$acOthIdx][0] == 155,
  "correctly finds C allele allele ac_oth value"
);
ok(
  $secondAllele->[$acMaleIdx][0] == 2860,
  "correctly finds C allele allele ac_male value"
);
ok(
  $secondAllele->[$acFemaleIdx][0] == 2372,
  "correctly finds C allele allele ac_female value"
);

ok(
  $secondAllele->[$anAfrIdx][0] == 7838,
  "correctly finds the C allele an_afr value, which has only one value across all alleles"
);
ok(
  $secondAllele->[$anAmrIdx][0] == 630,
  "correctly finds the C allele an_amr value, which has only one value across all alleles"
);
ok(
  $secondAllele->[$anAsjIdx][0] == 216,
  "correctly finds the C allele an_asj value, which has only one value across all alleles"
);
ok(
  $secondAllele->[$anEasIdx][0] == 1372,
  "correctly finds the C allele an_eas value, which has only one value across all alleles"
);
ok(
  $secondAllele->[$anFinIdx][0] == 2596,
  "correctly finds the C allele an_fin value, which has only one value across all alleles"
);
ok(
  $secondAllele->[$anNfeIdx][0] == 12638,
  "correctly finds the C allele an_nfe value, which has only one value across all alleles"
);
ok(
  $secondAllele->[$anOthIdx][0] == 776,
  "correctly finds the C allele an_oth value, which has only one value across all alleles"
);
ok(
  $secondAllele->[$anMaleIdx][0] == 14358,
  "correctly finds the C allele an_male value, which has only one value across all alleles"
);
ok(
  $secondAllele->[$anFemaleIdx][0] == 11708,
  "correctly finds the C allele an_female value, which has only one value across all alleles"
);

ok( $secondAllele->[$afAfrIdx][0] == unpack( 'f', pack( 'f', 1.52462e-01 ) ),
  "correctly finds the C allele af_afr value" );
ok( $secondAllele->[$afAmrIdx][0] == unpack( 'f', pack( 'f', 3.15873e-01 ) ),
  "correctly finds the C allele af_amr value" );
ok( $secondAllele->[$afAsjIdx][0] == unpack( 'f', pack( 'f', 2.22222e-01 ) ),
  "correctly finds the C allele af_asj value" );
ok( $secondAllele->[$afEasIdx][0] == unpack( 'f', pack( 'f', 3.36735e-01 ) ),
  "correctly finds the C allele af_eas value" );
ok( $secondAllele->[$afFinIdx][0] == unpack( 'f', pack( 'f', 2.07627e-01 ) ),
  "correctly finds the C allele af_fin value" );
ok( $secondAllele->[$afNfeIdx][0] == unpack( 'f', pack( 'f', 2.08419e-01 ) ),
  "correctly finds the C allele af_nfe value" );
ok( $secondAllele->[$afOthIdx][0] == unpack( 'f', pack( 'f', 1.99742e-01 ) ),
  "correctly finds the C allele af_oth value" );
ok( $secondAllele->[$afMaleIdx][0] == unpack( 'f', pack( 'f', 1.99192e-01 ) ),
  "correctly finds the C allele af_male value" );
ok( $secondAllele->[$afFemaleIdx][0] == unpack( 'f', pack( 'f', 2.02597e-01 ) ),
  "correctly finds the C allele af_female value" );

# this allele is found in the 2nd file as well
# this tests whether concurrency is broken by having 2 updates issued from one process
# and only 1 from another process, at roughly the same time
my $thirdAllele = [];
$href = $db->dbReadOne( 'chr22', 15927834 - 1 );
$vcf->get( $href, 'chr22', 'G', 'A', 0, $thirdAllele );

ok(
  $thirdAllele->[$trTvIdx][0] == 1, "trTv is 0 for multiallelics. \
  However, when our vcf parser is passed 2 alleles at the same position, but on different source file lines, it treats that as a SNP.
  Therefore it calls it as a G->A transition, or 1"
);
ok( $thirdAllele->[$idIdx][0] eq 'rsFake', "correctly finds the rsid" );
ok( $thirdAllele->[$acIdx][0] == 5232,
  "correctly finds the ac value for a biallelic site" );
ok( $thirdAllele->[$afIdx][0] == unpack( 'f', pack( 'f', 2.00721e-01 ) ),
  "correctly finds the af value for a biallelic site" );
ok( $thirdAllele->[$anIdx][0] == 26066,
  "correctly finds the an value for a biallelic site" );

ok(
  $thirdAllele->[$acAfrIdx][0] == 1195,
  "correctly finds the C allele ac_afr value"
);
ok( $thirdAllele->[$acAmrIdx][0] == 199,
  "correctly finds C allele allele ac_amr value" );
ok( $thirdAllele->[$acAsjIdx][0] == 48,
  "correctly finds C allele allele ac_asj value" );
ok( $thirdAllele->[$acEasIdx][0] == 462,
  "correctly finds C allele allele ac_eas value" );
ok( $thirdAllele->[$acFinIdx][0] == 539,
  "correctly finds C allele allele ac_fin value" );
ok(
  $thirdAllele->[$acNfeIdx][0] == 2634,
  "correctly finds C allele allele ac_nfe value"
);
ok( $thirdAllele->[$acOthIdx][0] == 155,
  "correctly finds C allele allele ac_oth value" );
ok(
  $thirdAllele->[$acMaleIdx][0] == 2860,
  "correctly finds C allele allele ac_male value"
);
ok(
  $thirdAllele->[$acFemaleIdx][0] == 2372,
  "correctly finds C allele allele ac_female value"
);

ok(
  $thirdAllele->[$anAfrIdx][0] == 7838,
  "correctly finds the C allele an_afr value, which has only one value across all alleles"
);
ok(
  $thirdAllele->[$anAmrIdx][0] == 630,
  "correctly finds the C allele an_amr value, which has only one value across all alleles"
);
ok(
  $thirdAllele->[$anAsjIdx][0] == 216,
  "correctly finds the C allele an_asj value, which has only one value across all alleles"
);
ok(
  $thirdAllele->[$anEasIdx][0] == 1372,
  "correctly finds the C allele an_eas value, which has only one value across all alleles"
);
ok(
  $thirdAllele->[$anFinIdx][0] == 2596,
  "correctly finds the C allele an_fin value, which has only one value across all alleles"
);
ok(
  $thirdAllele->[$anNfeIdx][0] == 12638,
  "correctly finds the C allele an_nfe value, which has only one value across all alleles"
);
ok(
  $thirdAllele->[$anOthIdx][0] == 776,
  "correctly finds the C allele an_oth value, which has only one value across all alleles"
);
ok(
  $thirdAllele->[$anMaleIdx][0] == 14358,
  "correctly finds the C allele an_male value, which has only one value across all alleles"
);
ok(
  $thirdAllele->[$anFemaleIdx][0] == 11708,
  "correctly finds the C allele an_female value, which has only one value across all alleles"
);

ok( $thirdAllele->[$afAfrIdx][0] == unpack( 'f', pack( 'f', 1.52462e-01 ) ),
  "correctly finds the C allele af_afr value" );
ok( $thirdAllele->[$afAmrIdx][0] == unpack( 'f', pack( 'f', 3.15873e-01 ) ),
  "correctly finds the C allele af_amr value" );
ok( $thirdAllele->[$afAsjIdx][0] == unpack( 'f', pack( 'f', 2.22222e-01 ) ),
  "correctly finds the C allele af_asj value" );
ok( $thirdAllele->[$afEasIdx][0] == unpack( 'f', pack( 'f', 3.36735e-01 ) ),
  "correctly finds the C allele af_eas value" );
ok( $thirdAllele->[$afFinIdx][0] == unpack( 'f', pack( 'f', 2.07627e-01 ) ),
  "correctly finds the C allele af_fin value" );
ok( $thirdAllele->[$afNfeIdx][0] == unpack( 'f', pack( 'f', 2.08419e-01 ) ),
  "correctly finds the C allele af_nfe value" );
ok( $thirdAllele->[$afOthIdx][0] == unpack( 'f', pack( 'f', 1.99742e-01 ) ),
  "correctly finds the C allele af_oth value" );
ok( $thirdAllele->[$afMaleIdx][0] == unpack( 'f', pack( 'f', 1.99192e-01 ) ),
  "correctly finds the C allele af_male value" );
ok( $thirdAllele->[$afFemaleIdx][0] == unpack( 'f', pack( 'f', 2.02597e-01 ) ),
  "correctly finds the C allele af_female value" );

# Let's see what happens if a user wants to show both alleles on the same line
my $multiallelic = [ [], [] ];

$href = $db->dbReadOne( 'chr22', 15927834 - 1 );

$vcf->get( $href, 'chr22', 'G', 'T', 0, $multiallelic->[0] );

$vcf->get( $href, 'chr22', 'G', 'C', 0, $multiallelic->[1] );

for my $alleleIdx ( 0 .. $#$multiallelic ) {
  for my $feature ( @{ $vcf->features } ) {
    my $featureIdx = $vcf->getFieldDbName($feature);
    my $posIdx     = 0;

    if ( $alleleIdx == 0 ) {
      if ( !defined $multiallelic->[$alleleIdx][$featureIdx][$posIdx] ) {
        ok(
          !defined $firstAllele->[$alleleIdx][$featureIdx][$posIdx],
          "multiallelics are reproduced just like bi-allelics, but on single line for feature "
            . $vcf->features->[$featureIdx]
        );
      }
      elsif ( looks_like_number( $multiallelic->[$featureIdx][$alleleIdx][$posIdx] ) ) {
        ok(
          $multiallelic->[$alleleIdx][$featureIdx][$posIdx]
            == $firstAllele->[$featureIdx][$posIdx],
          "multiallelics are reproduced just like bi-allelics, but on single line for feature "
            . $vcf->features->[$featureIdx]
        );
      }
      else {
        ok(
          $multiallelic->[$alleleIdx][$featureIdx][$posIdx] eq
            $firstAllele->[$featureIdx][$posIdx],
          "multiallelics are reproduced just like bi-allelics, but on single line for feature "
            . $vcf->features->[$featureIdx]
        );
      }

      next;
    }

    if ( $alleleIdx == 1 ) {
      if ( !defined $multiallelic->[$alleleIdx][$featureIdx][$posIdx] ) {
        ok(
          !defined $secondAllele->[$featureIdx][$posIdx],
          "multiallelics are reproduced just like bi-allelics, but on single line for feature "
            . $vcf->features->[$featureIdx]
        );
      }
      elsif ( looks_like_number( $multiallelic->[$alleleIdx][$featureIdx][$posIdx] ) ) {
        ok(
          $multiallelic->[$alleleIdx][$featureIdx][$posIdx]
            == $secondAllele->[$featureIdx][$posIdx],
          "multiallelics are reproduced just like bi-allelics, but on single line for feature "
            . $vcf->features->[$featureIdx]
        );
      }
      else {
        ok(
          $multiallelic->[$alleleIdx][$featureIdx][$posIdx] eq
            $secondAllele->[$featureIdx][$posIdx],
          "multiallelics are reproduced just like bi-allelics, but on single line for feature "
            . $vcf->features->[$featureIdx]
        );
      }
    }
  }
}

$db->cleanUp();
done_testing();
