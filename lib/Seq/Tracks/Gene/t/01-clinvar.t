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

my $mock = MockBuild->new_with_config({
  config => './lib/Seq/Tracks/Gene/t/clinvar-test-config.yml',
  chromosomes => ['chrY'],
  build_region_track_only => 1,
  verbose => 0,
});

my $trackBuilder = $mock->tracksObj->getTrackBuilderByName('refSeq');

$trackBuilder->buildTrack();

# $mock = MockBuild->new_with_config({
#   config => './lib/Seq/Tracks/Gene/t/clinvar-test-config.yml',
#   chromosomes => ['chrY'],
# });

# my $refTrackBuilder = $mock->tracksObj->getTrackBuilderByName('ref');

# $refTrackBuilder->buildTrack();

# $trackBuilder = $mock->tracksObj->getTrackBuilderByName('refSeq');

# $trackBuilder->buildTrack();

my $refSeqTrackGetter = $mock->tracksObj->getTrackGetterByName('refSeq');

my $featuresAref = $refSeqTrackGetter->features;


my $db = Seq::DBManager->new();

my $dataAref = $db->dbReadOne('chrY', 7273972 - 1);

p $dataAref;
my $refSeqTrackIdx = $refSeqTrackGetter->dbName;

# p $clinvarTrackIndex;

my $regionTrackData = $db->dbReadAll($refSeqTrackGetter->regionTrackPath('chrY'));

my $names = $refSeqTrackGetter->{_allCachedDbNames};

# The features we're testing
# features:
#   - kgID 0
#   - mRNA 1
#   - spID 2
#   - spDisplayID 3
#   - refseq 4
#   - protAcc 5
#   - description 6
#   - rfamAcc 7
#   - name 8
#   - name2 9
#   fetch_date: 2017-02-09T17:07:00
#   join:
#     features:
#     - alleleID 10
#     - phenotypeList 11
#     - clinicalSignificance 12
#     - reviewStatus 13
#     - type 14
#     - chromStart 15
#     - chromEnd 16
    # track: clinvar

# The full refGene transcript file we're testing
# bin     name    chrom   strand  txStart txEnd   cdsStart        cdsEnd  exonCount       exonStarts      exonEnds        score   name2   cdsStartStat    cdsEndStat      exonFrames      kgID    mRNA    spID    spDisplayID     geneSymbol      refseq  protAcc description     rfamAcc tRnaName
# 9       NR_028062       chrY    +       7273971 7381547 7381547 7381547 8       7273971,7303936,7325905,7341114,7356134,7367355,7371742,7375711,        7274478,7304105,7326169,7341234,7356230,7367433,7371889,7381547,        0       PRKY    unk     unk     -1,-1,-1,-1,-1,-1,-1,-1,        NA      NA      NA      NA      NA      NA      NA      NA      NA      NA
# 10      NM_001206850    chrY    +       14522607        14843968        14723148        14841262        6       14522607,14719458,14723116,14824187,14829729,14840412,  14522941,14719518,14723269,14824373,14830519,14843968,  0       NLGN4Y  cmpl    cmpl    -1,-1,0,1,1,2,  uc004fte.3      NM_001206850    Q8NFZ3  NLGNY_HUMAN     NLGN4Y  NM_001206850    NM_001206850    Homo sapiens neuroligin 4, Y-linked (NLGN4Y), transcript variant 3, mRNA. (from RefSeq NM_001206850)    NA      NA
# 10      NR_046355       chrY    +       14523504        14843968        14843968        14843968        7       14523504,14622008,14719458,14723116,14824187,14829729,14840412, 14523572,14622591,14719518,14723269,14824373,14830519,14843968, 0       NLGN4Y  unk     unk     -1,-1,-1,-1,-1,-1,-1,   NA      NA      NA      NA      NA      NA      NA      NA      NA      NA
# 10      NM_014893       chrY    +       14523745        14843968        14622119        14841262        6       14523745,14622020,14723116,14824187,14829729,14840412,  14523898,14622591,14723269,14824373,14830519,14843968,  0       NLGN4Y  cmpl    cmpl    -1,0,1,1,1,2,   uc004ftg.3      NM_014893       Q8NFZ3  NLGNY_HUMAN     NLGN4Y  NM_014893       NM_014893       Homo sapiens neuroligin 4, Y-linked (NLGN4Y), transcript variant 1, mRNA. (from RefSeq NM_014893)       NA      NA
# 10      NR_028319       chrY    +       14524573        14843968        14843968        14843968        6       14524573,14622008,14723116,14824187,14829729,14840412,  14524936,14622591,14723269,14824373,14830519,14843968,  0       NLGN4Y  unk     unk     -1,-1,-1,-1,-1,-1,      NA      NA      NA      NA      NA      NA      NA      NA      NA      NA
# 10      NM_001164238    chrY    +       14622020        14733549        14622119        14733537        4       14622020,14719458,14723116,14733451,    14622591,14719518,14723269,14733549,    0       NLGN4Y  cmpl    cmpl    0,1,1,1,        uc004fti.4      NM_001164238    Q8NFZ3  NLGNY_HUMAN     NLGN4Y  NM_001164238    NM_001164238    Homo sapiens neuroligin 4, Y-linked (NLGN4Y), transcript variant 2, mRNA. (from RefSeq NM_001164238)    NA      NA
# 11      NR_125736       chrY    -       18872500        19076003        19076003        19076003        5       18872500,18877067,18912367,19068723,19075940,   18872834,18877205,18912482,19068798,19076003,   0       TTTY14  unk     unk     -1,-1,-1,-1,-1, NA      NA      NA      NA      NA      NA      NA      NA      NA      NA
# 11      NR_125733       chrY    -       18872500        19077547        19077547        19077547        5       18872500,18932742,18933356,19068723,19077044,   18872834,18932841,18933475,19068798,19077547,   0       TTTY14  unk     unk     -1,-1,-1,-1,-1, NA      NA      NA      NA      NA      NA      NA      NA      NA      NA
# 11      NR_125734       chrY    -       18872500        19077547        19077547        19077547        4       18872500,18877067,19068723,19077044,    18872834,18877205,19068798,19077547,    0       TTTY14  unk     unk     -1,-1,-1,-1,    NA      NA      NA      NA      NA      NA      NA      NA      NA      NA

# The full clinvar file we're testing
#AlleleID       Type    Name    GeneID  GeneSymbol      HGNC_ID ClinicalSignificance    ClinSigSimple   LastEvaluated   RS# (dbSNP)     nsv/esv (dbVar) RCVaccession    PhenotypeIDS    PhenotypeList   Origin  OriginSimple    Assembly        ChromosomeAccession     Chromosome      Start   Stop    ReferenceAllele AlternateAllele Cytogenetic     ReviewStatus    NumberSubmitters        Guidelines      TestedInGTR     OtherIDs        SubmitterCategories
# 24776   deletion        NM_003140.2(SRY):c.364_367delGAGA (p.Glu122Asnfs)       6736    SRY     HGNC:11311      Pathogenic      1       Jan 01, 1993    606231178       -       RCV000010390    MedGen:C2748896,OMIM:400044     46,XY sex reversal, type 1 (DELETION REFSEQ OVERLAP TEST)       germline        germline        GRCh38  NC_000024.10    Y       7273971 7273974 TCTC    -       Yp11.2  no assertion criteria provided  1               N       OMIM Allelic Variant:480000.0001        1
# 99999   single nucleotide variant       NM_003140.2(SRY):c.326T>C (p.Phe109Ser) 6736    SRY     HGNC:11311      Pathogenic      1       Dec 01, 1992    104894956       -       RCV000010391    MedGen:C2748896,OMIM:400044     46,XY sex reversal, type 1 (SNP REFSEQ OVERLAP TEST)    germline        germline        GRCh38  NC_000024.9     Y       14523504        14523504 A      G       Yp11.3  no assertion criteria provided  1               N       OMIM Allelic Variant:480000.0003,UniProtKB (protein):Q05066#VAR_003730_FAKE_9999        1
# 24777   single nucleotide variant       NM_003140.2(SRY):c.326T>C (p.Phe109Ser) 6736    SRY     HGNC:11311      Pathogenic      1       Dec 01, 1992    104894956       -       RCV000010391    MedGen:C2748896,OMIM:400044     46,XY sex reversal, type 1 (OVERLAP CLINVAR REFSEQ OVERLAP TEST)        germline        germline        GRCh38  NC_000024.9     Y       7273972 7273972 A       G       Yp11.3  no assertion criteria provided  1               N       OMIM Allelic Variant:480000.0003,UniProtKB (protein):Q05066#VAR_003730  1

#You'll notice that the first clinvar entry is at 7273971 to 7273974
# This clearly overlaps the first transcript, NR_028062, which also overlaps no other transcripts
# However, the last entry 24777, also overlaps this NR_028062 transcript
my $txData = $regionTrackData->{0};
p $txData;
ok ($txData->{8} eq "NR_028062");
ok($txData->{9} eq "PRKY");
ok($txData->{10}[0] == 24776);
ok($txData->{10}[1] == 24777);
ok($txData->{11}[0] eq "46,XY sex reversal, type 1 (DELETION REFSEQ OVERLAP TEST)");
ok($txData->{11}[1] eq "46,XY sex reversal, type 1 (OVERLAP CLINVAR REFSEQ OVERLAP TEST)");
ok($txData->{12}[0] eq "Pathogenic");
ok($txData->{12}[1] eq "Pathogenic");
ok($txData->{13}[0] eq "no assertion criteria provided");
ok($txData->{13}[1] eq "no assertion criteria provided");
ok($txData->{14}[0] eq "deletion");
ok($txData->{14}[1] eq "single nucleotide variant");
ok($txData->{15}[0] == 7273971);
ok($txData->{15}[1] == 7273972);
ok($txData->{16}[0] == 7273974);
ok($txData->{16}[1] == 7273972);
done_testing();
1;