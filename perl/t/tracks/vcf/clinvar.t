use 5.10.0;
use strict;
use warnings;

package MockBuilder;
use lib './lib';
use Mouse;
extends 'Seq::Base';

1;

use Seq::Tracks::Reference::Build;
use Seq::Tracks::Reference::MapBases;
use Seq::Tracks::Vcf::Build;
use YAML::XS qw/LoadFile/;

use Test::More;
use Path::Tiny;
use Scalar::Util qw/looks_like_number/;
use DDP;

my $baseMapper = Seq::Tracks::Reference::MapBases->new();

my $file   = LoadFile('./t/tracks/vcf/clinvar.yml');
my $dbPath = $file->{database_dir};

path($dbPath)->remove_tree( { keep_root => 1 } );

my $seq = MockBuilder->new_with_config(
  { config => path('./t/tracks/vcf/clinvar.yml')->absolute, debug => 0 } );

my $tracks     = $seq->tracksObj;
my $refBuilder = $tracks->getRefTrackBuilder();
my $refGetter  = $tracks->getRefTrackGetter();

$refBuilder->db->dbPatch(
  'chr1', $refBuilder->dbName,
  1022260 - 1,
  $baseMapper->baseMap->{'C'}
); #chr1:1022260

$refBuilder->db->cleanUp();

my $dbVar = $refBuilder->db->dbReadOne( 'chr1', 1022260 - 1 );
ok( $refGetter->get($dbVar) eq 'C' );

my $vcfBuilder = $tracks->getTrackBuilderByName('clinvar.match');

$vcfBuilder->buildTrack();

my $vcf = $tracks->getTrackGetterByName('clinvar.match');

my $db = Seq::DBManager->new();

############### Feature tests ################
# The vcf file contains the following items:
# 1       1022260 .       C       T       .       .       START=1022260;STOP=1022260;STRAND=+;VARIATION_TYPE=Variant;VARIATION_ID=128296;RCV=RCV000116258|RCV000550396;SCV=SCV000317028|SCV000150176|SC
# V000653894;ALLELE_ID=133745;SYMBOL=AGRN;HGVS_C=NM_198576.3:c.261C>T;HGVS_P=NP_940978.2:p.Asp87_eq_;MOLECULAR_CONSEQUENCE=NM_198576.3:c.261C>T:synonymous_variant;CLINICAL_SIGNIFICANCE=Benign/Likely_
# benign;CLINICAL_SIGNIFICANCE_ORDERED=likely_benign|benign;PATHOGENIC=0;LIKELY_PATHOGENIC=0;UNCERTAIN_SIGNIFICANCE=0;LIKELY_BENIGN=1;BENIGN=2;REVIEW_STATUS=criteria_provided..multiple_submitters..no
# _conflicts;REVIEW_STATUS_ORDERED=no_assertion_criteria_provided|criteria_provided..single_submitter;LAST_EVALUATED=Aug_11..2017;ALL_SUBMITTERS=PreventionGenetics|Genetic_Services_Laboratory..Univer
# sity_of_Chicago|Invitae;SUBMITTERS_ORDERED=Genetic_Services_Laboratory..University_of_Chicago|PreventionGenetics|Invitae;ALL_TRAITS=not_specified|AllHighlyPenetrant|NOT_SPECIFIED|Myasthenic_syndrom
# e..congenital..8;ALL_PMIDS=25741868|20301347|28492532;ORIGIN=germline;XREFS=MedGen:CN169374|GeneReviews:NBK1168|MedGen:C3808739|OMIM:615120|Orphanet:590;DATES_ORDERED=0000-00-00|2017-08-11

my $href = $db->dbReadOne( 'chr1', 1022260 - 1 );

#my ($vcf, $href, $chr, $refBase, $allele, $alleleIdx, $positionIdx, $outAccum) = @_;
# At this position we have CACA,G alleles
my $out = [];

# Vcf tracks are going to be treated differently from transcripts and sparse tracks
# They should return nothing for the nth position in a deletion
# Or if the allele doesn't match exactly.
$vcf->get( $href, 'chr1', 'C', 'G', 0, $out );

ok( @{$out} == 0, "Alleles that don't match won't get reported" );

$out = [];
$vcf->get( $href, 'chr1', 'C', 'T', 0, $out );
p $out;

ok( @{$out} == @{ $vcf->features },
  "Report matching alleles, with one entry per requested feature" );

my $vidIdx        = $vcf->getFieldDbName('variation_id');
my $aidIdx        = $vcf->getFieldDbName('allele_id');
my $rcvIdx        = $vcf->getFieldDbName('rcv');
my $scvIdx        = $vcf->getFieldDbName('scv');
my $hgvscIdx      = $vcf->getFieldDbName('hgvs_c');
my $hgvspIdx      = $vcf->getFieldDbName('hgvs_p');
my $molIdx        = $vcf->getFieldDbName('molecular_consequence');
my $submittersIdx = $vcf->getFieldDbName('all_submitters');
my $datesIdx      = $vcf->getFieldDbName('dates');
my $pathIdx       = $vcf->getFieldDbName('pathogenic');
my $lpathIdx      = $vcf->getFieldDbName('likely_pathogenic');
my $ucIdx         = $vcf->getFieldDbName('uncertain_significance');
my $benignIdx     = $vcf->getFieldDbName('benign');
my $lbenignIdx    = $vcf->getFieldDbName('likely_benign');
my $rsIdx         = $vcf->getFieldDbName('review_status');

# we specify "ALL_TRAITS" => 'traits' in YAML
my $traitsIdx = $vcf->getFieldDbName('traits');

my $numFeatures = scalar @{ $vcf->features };
ok( @{$out} == $numFeatures,
  "vcf array contains an entry for each requested feature" );

ok( $out->[$vidIdx][0] == 128296, 'can reproduce variant_id' );
ok( $out->[$aidIdx][0] == 133745, 'can reproduce allele_id' );

ok( $out->[$rcvIdx][0][0] eq 'RCV000116258',
  'can reproduce first rcv, by splitting on pipe as defined in YAML' );
ok( $out->[$rcvIdx][0][1] eq 'RCV000550396',
  'can reproduce second rcv, by splitting on pipe as defined in YAML' );

ok( $out->[$scvIdx][0][0] eq 'SCV000317028',
  'can reproduce first scv, by splitting on pipe as defined in YAML' );
ok( $out->[$scvIdx][0][1] eq 'SCV000150176',
  'can reproduce second scv, by splitting on pipe as defined in YAML' );
ok( $out->[$scvIdx][0][2] eq 'SCV000653894',
  'can reproduce third scv, by splitting on pipe as defined in YAML' );

ok(
  !defined $out->[$traitsIdx][0][0],
  'traits are split by pipe as specified in YAML, and not_specified is cleaned to undef in the array'
);
ok(
  $out->[$traitsIdx][0][1] eq 'AllHighlyPenetrant',
  'traits are split by pipe as specified in YAML, finding AllHighlyPenetrant in the correct order'
);
ok(
  !defined $out->[$traitsIdx][0][2],
  'traits are split by pipe as specified in YAML, and NOT_SPECIFIED is cleaned to undef in the array'
);
ok(
  $out->[$traitsIdx][0][3] eq 'Myasthenic_syndrome..congenital..8',
  'traits are split by pipe as specified in YAML, finding Myasthenic_syndrome..congenital..8 in the correct order'
);

$db->cleanUp();

path($dbPath)->remove_tree( { keep_root => 1 } );
done_testing();
