use 5.10.0;
use strict;
use warnings;
use lib './lib';


package MockBuild;
use Mouse 2;
use Seq::Tracks::Build;
extends "Seq::Base";
#TODO: allow building just one track, identified by name
has config => (is => 'ro', isa => 'Str', required => 1);

package MockAnnotate;
use lib './lib';
use Test::More;
use DDP;
use Seq::DBManager;
use Seq;

my $mock = MockBuild->new_with_config({config => './lib/Seq/Tracks/Sparse/t/clinvar-test-config.yml', chromosomes => ['chrY'], verbose => 0 });

my $trackBuilder = $mock->tracksObj->getTrackBuilderByName('clinvar');

p $trackBuilder;

$trackBuilder->buildTrack();

my $clinvarTrackGetter = $mock->tracksObj->getTrackGetterByName('clinvar');

# The config file has these features
  # - alleleID: number
  # - phenotypeList
  # - clinicalSignificance
  # - type
  # - origin
  # - numberSubmitters
  # - reviewStatus
  # - referenceAllele
  # - alternateAllele


my $inputAref = [['chr22', 14000001, 'A', 'G', 'SNP', 'G', 1]];

my $db = Seq::DBManager->new();

# We expect an overlap, first the indel, then the snp
my $dataAref = $db->dbReadOne('chrY', 2787237 - 1);

# p $dataAref;

my $clinvarTrackIndex = $clinvarTrackGetter->dbName;

# p $clinvarTrackIndex;

my $clinvarDataAref = $dataAref->[$clinvarTrackIndex];
my @clinvarData = @$clinvarDataAref;
p @clinvarData;

ok($clinvarData[0][0] == 24776 && $clinvarData[0][1] == 99999);
ok($clinvarData[1][0] eq "46,XY sex reversal, type 1" && $clinvarData[1][1] eq "46,XY sex reversal, type 1 (FAKE TO TEST OVERLAP)");
ok($clinvarData[2][0] eq "Pathogenic" && $clinvarData[2][1] eq "Pathogenic");
ok($clinvarData[3][0] eq "deletion" && $clinvarData[3][1] eq "single nucleotide variant");
ok($clinvarData[4][0] eq "germline" && $clinvarData[4][1] eq "germline");
#numberSubmitters
ok($clinvarData[5][0] == 1 && $clinvarData[5][1] == 1);
ok($clinvarData[6][0] eq "no assertion criteria provided" && $clinvarData[6][1] eq "no assertion criteria provided");
ok($clinvarData[7][0] eq "TCTC" && $clinvarData[7][1] eq "A");
ok($clinvarData[8][0] eq "-" && $clinvarData[8][1] eq "G");

# We expect an overlap, first the indel, then the snp
$dataAref = $db->dbRead('chrY', [2787238 - 1 .. 2787240 -1] );

# Deletions spans all deleted bases (this one goes from 2787237 to 2787240 in 1-base notation)

for my $data (@$dataAref) {
  $clinvarDataAref = $data->[$clinvarTrackIndex];
  @clinvarData = @$clinvarDataAref;
 p @clinvarData;
  ok($clinvarData[0] == 24776 );
  ok($clinvarData[1] eq "46,XY sex reversal, type 1");
  ok($clinvarData[2] eq "Pathogenic");
  ok($clinvarData[3] eq "deletion");
  ok($clinvarData[4] eq "germline");
  #numberSubmitters
  ok($clinvarData[5] == 1);
  ok($clinvarData[6] eq "no assertion criteria provided");
  ok($clinvarData[7] eq "TCTC");
  ok($clinvarData[8] eq "-");
}

$dataAref = $db->dbReadOne('chrY', 2787334 - 1);

$clinvarDataAref = $dataAref->[$clinvarTrackIndex];
@clinvarData = @$clinvarDataAref;
p @clinvarData;

ok($clinvarData[0] == 24780 );
ok($clinvarData[1][0] eq "46,XY sex reversal, type 1" && $clinvarData[1][1] eq "46,XY true hermaphroditism, SRY-related");
ok($clinvarData[2] eq "Pathogenic");
ok($clinvarData[3] eq "single nucleotide variant");
ok($clinvarData[4] eq "germline");
#numberSubmitters
ok($clinvarData[5] == 1);
ok($clinvarData[6] eq "no assertion criteria provided");
ok($clinvarData[7] eq "G");
ok($clinvarData[8] eq "C");

done_testing();
1;