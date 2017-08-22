use 5.10.0;
use strict;
use warnings;

package MockBuilder;
use lib './lib';
use Mouse;
use DDP;
extends 'Seq::Base';

1;

use Seq::Tracks::Reference::Build;
use Seq::Tracks::Reference::MapBases;
use Seq::Tracks::Vcf::Build;

use Test::More;
use Path::Tiny;

my $baseMapper = Seq::Tracks::Reference::MapBases->new();


my $seq = MockBuilder->new_with_config({config => path('./t/tracks/vcf/test.hg38.chr22.yml')->absolute, debug => 1});

my $tracks = $seq->tracksObj;
my $refBuilder = $tracks->getRefTrackBuilder();
my $refGetter = $tracks->getRefTrackGetter();
p $refBuilder;

$refBuilder->db->dbPatch('chr22', $refBuilder->dbName, 15927888 - 1, $baseMapper->baseMap->{'C'}); #chr14:19792736-19792737 #same
$refBuilder->db->dbPatch('chr22', $refBuilder->dbName, 15927876 - 1, $baseMapper->baseMap->{'G'}); #chr14:19792727 #same
$refBuilder->db->dbPatch('chr22', $refBuilder->dbName, 15927837 - 1, $baseMapper->baseMap->{'A'}); #chr14:19792869-19792870 #same
$refBuilder->db->dbPatch('chr22', $refBuilder->dbName, 15927835 - 1, $baseMapper->baseMap->{'G'}); #chr14:19792857-19792858 #same
$refBuilder->db->dbPatch('chr22', $refBuilder->dbName, 15927834 - 1, $baseMapper->baseMap->{'G'}); #chr14:19792818-19792819 #same
$refBuilder->db->dbPatch('chr22', $refBuilder->dbName, 15927765 - 1, $baseMapper->baseMap->{'A'}); #chr14:19792816-19792817 #same
$refBuilder->db->dbPatch('chr22', $refBuilder->dbName, 15927759 - 1, $baseMapper->baseMap->{'A'}); #chr14:19792815-19792816 #same
$refBuilder->db->dbPatch('chr22', $refBuilder->dbName, 15927755 - 1, $baseMapper->baseMap->{'T'}); #On #chr14:19792746-19792747 #same
$refBuilder->db->dbPatch('chr22', $refBuilder->dbName, 15927745 - 1, $baseMapper->baseMap->{'A'}); #chr14:19792740-19792741 #same

$refBuilder->db->cleanUp();

my $dbVar = $refBuilder->db->dbReadOne('chr22', 15927888 - 1);
ok($refGetter->get($dbVar) eq 'C');

$dbVar = $refBuilder->db->dbReadOne('chr22', 15927876 - 1);
ok($refGetter->get($dbVar) eq 'G');

$dbVar = $refBuilder->db->dbReadOne('chr22', 15927837 - 1);
ok($refGetter->get($dbVar) eq 'A');

$dbVar = $refBuilder->db->dbReadOne('chr22', 15927835 - 1);
ok($refGetter->get($dbVar) eq 'G');

$dbVar = $refBuilder->db->dbReadOne('chr22', 15927834 - 1);
ok($refGetter->get($dbVar) eq 'G');

$dbVar = $refBuilder->db->dbReadOne('chr22', 15927765 - 1);
ok($refGetter->get($dbVar) eq 'A');

$dbVar = $refBuilder->db->dbReadOne('chr22', 15927759 - 1);
ok($refGetter->get($dbVar) eq 'A');

$dbVar = $refBuilder->db->dbReadOne('chr22', 15927755 - 1); #on
ok($refGetter->get($dbVar) eq 'T');

$dbVar = $refBuilder->db->dbReadOne('chr22', 15927745 - 1);
ok($refGetter->get($dbVar) eq 'A');

my $vcfBuilder = $tracks->getTrackBuilderByName('gnomad.genomes');

$vcfBuilder->buildTrack();

my $vcf = $tracks->getTrackGetterByName('gnomad.genomes');

my $db = Seq::DBManager->new();

my $href = $db->dbReadOne('chr22', 15927888 - 1);

#my ($self, $href, $chr, $refBase, $allele, $alleleIdx, $positionIdx, $outAccum) = @_;
# At this position we have CACA,G alleles
my $out = [];

$vcf->get($href, 'chr22', 'C', 'G', 0, 0, $out);

p $out;
# say "test is $test";