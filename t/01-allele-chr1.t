use 5.10.0;
use warnings;
use strict;

# TODO: For this test suite, we expect that the gene track properly
# stores codonSequence, codonNumber, codonPosition

package MockAnnotationClass;
use lib './lib';
use Mouse;
extends 'Seq';
use Seq::Tracks;
use DDP;


# For this test, not used
has '+input_file' => (default => 'test.snp');

# output_file_base contains the absolute path to a file base name
# Ex: /dir/child/BaseName ; BaseName is appended with .annotated.tab , .annotated-log.txt, etc
# for the various outputs
has '+output_file_base' => (default => 'test');

has trackIndices => (is => 'ro', isa => 'HashRef', init_arg => undef, writer => '_setTrackIndices');
has trackFeatureIndices => (is => 'ro', isa => 'HashRef', init_arg => undef, writer => '_setTrackFeatureIndices');

sub BUILD {
  my $self = shift;
  $self->{_chrFieldIdx} = 0;
  $self->{_positionFieldIdx} = 1;
  $self->{_referenceFieldIdx} = 2;
  $self->{_alleleFieldIdx} = 3;
  $self->{_typeFieldIdx} = 4;

  my $headers = Seq::Headers->new();

  my %trackIdx = %{ $headers->getParentFeaturesMap() };

  $self->_setTrackIndices(\%trackIdx);

  my %childFeatureIndices;

  for my $trackName (keys %trackIdx ) {
    $childFeatureIndices{$trackName} = $headers->getChildFeaturesMap($trackName);
  }

  $self->_setTrackFeatureIndices(\%childFeatureIndices);
}

1;

package TestRead;
use DDP;
use lib './lib';
use Test::More;
use Seq::DBManager;
use MCE::Loop;
use Utils::SqlWriter::Connection;
use Seq::Headers;
use List::Util qw/first/;
use List::MoreUtils qw/first_index/;

my $genome = 'hg38';
my $chr = "chr100";

system('touch test.snp');

my $annotator = MockAnnotationClass->new_with_config({ config => "./config/$genome.yml"});
my $db = Seq::DBManager->new();

my $numEntries = $db->dbGetNumberOfEntries($chr);
if( !$numEntries ) {
  plan skip_all => "No database for $genome, $chr found";
}









my $sqlClient = Utils::SqlWriter::Connection->new();

my $dbh = $sqlClient->connect('hg38');

my $tracks = Seq::Tracks->new({tracks => $annotator->tracks, gettersOnly => 1});


# Set the lmdb database to read only, remove locking
# We MUST make sure everything is written to the database by this point
$db->setReadOnly(1);
  
my $geneTrack = $tracks->getTrackGetterByName('refSeq');

my $dataHref;

my $geneTrackIdx = $geneTrack->dbName;
my $nearestTrackIdx = $geneTrack->nearestDbName;

my @allChrs = $geneTrack->allWantedChrs();

my $chr = 'chr22';

# plan tests => 4;

my $geneTrackRegionData = $db->dbReadAll( $geneTrack->regionTrackPath($chr) );


#Using the above defined
# $self->{_chrFieldIdx} = 0;
# $self->{_positionFieldIdx} = 1;
# $self->{_referenceFieldIdx} = 2;
# $self->{_alleleFieldIdx} = 3;
# $self->{_typeFieldIdx} = 4;
# 1st and only smaple genotype = 5;
# 1st and only sample genotype confidence = 6

# Seq.pm sets these in the annotate method, based on the input file
# Here we mock that file.

my $inputAref;

my $dataAref;

my $outAref = [];

my @outData;

my $trackIndices = $annotator->trackIndices;

my $geneTrackData = $outData[$trackIndices->{refSeq}];

my $headers = Seq::Headers->new();

my @allGeneTrackFeatures = @{ $headers->getParentFeatures($geneTrack->name) };

say "all gene track features";
p @allGeneTrackFeatures;

say "\nBeginning tests\n";

my %geneTrackFeatureMap;
# This includes features added to header, using addFeatureToHeader 
# such as the modified nearest feature names ($nTrackPrefix.$_) and join track names
# and siteType, strand, codonNumber, etc.
for my $i (0 .. $#allGeneTrackFeatures) {
  $geneTrackFeatureMap{ $allGeneTrackFeatures[$i] } = $i;
}
my $sth = $dbh->prepare('SELECT * FROM hg38.refGene WHERE chrom="chr22" AND (txStart <= 14000001 AND txEnd>=14000001) OR (txStart >= 14000001 AND txEnd<=14000001)');

$sth->execute();

#Schema:
#   0 , 1   , 2    , 3      ... 12
# [bin, name, chrom, strand, ...name2,
# 
my @row = $sth->fetchrow_array;

ok(!@row, "UCSC still has 14000001 as intergenic");
# ok($geneTrackData->[0][0][0] eq 'intergenic', "We have this as intergenic");


# Check a site with 1 transcript, on the negative strand



$geneTrackData = $outData[$trackIndices->{refSeq}];

$sth = $dbh->prepare("SELECT * FROM hg38.refGene WHERE chrom='chr22' AND (txStart <= 45950000 AND txEnd>=45950000) OR (txStart >= 45950000 AND txEnd<=45950000);");

$sth->execute();

@row = $sth->fetchrow_array;

# TODO: move away from requiring ref name.
my $refTrackIdx = $trackIndices->{ref};
my $txNameIdx = $geneTrackFeatureMap{refseq};
my $geneSymbolIdx = $geneTrackFeatureMap{name2};
my $refAAidx = $geneTrackFeatureMap{$geneTrack->refAminoAcidKey};
my $altAAidx = $geneTrackFeatureMap{$geneTrack->newAminoAcidKey};
my $refCodonIdx = $geneTrackFeatureMap{$geneTrack->codonSequenceKey};
my $altCodonIdx = $geneTrackFeatureMap{$geneTrack->newCodonKey};
my $strandIdx = $geneTrackFeatureMap{$geneTrack->strandKey};
my $codonPositionIdx = $geneTrackFeatureMap{$geneTrack->codonPositionKey};
my $codonNumberIdx = $geneTrackFeatureMap{$geneTrack->codonNumberKey};

my $exonicAlleleFunctionIdx = $geneTrackFeatureMap{$geneTrack->exonicAlleleFunctionKey};
my $siteTypeIdx = $geneTrackFeatureMap{$geneTrack->siteTypeKey};


my $header = ['Fragment', 'Position', 'Reference', 'Allele', 'Type', 'Sample_4', '', 'Sample_5', ''];
$annotator->{_inputHeader} = $header;
$annotator->{_sampleGenosIdx} = [5, 7];

say "Teting chr1 : 40,074,058 which had a clinvar phenotypeids record in old seqnat";
$inputAref = [['chr1', 40074058, 'G', 'T', 'SNP', 'G', 1, 'T', 1]];
$outAref = $annotator->addTrackData($inputAref);
p $outAref;

say "\n\nTesting a synonymous stop site SNP  on the negative strand of chr1:115286071\n";

# deleted 45950143 - 45950148
#http://genome.ucsc.edu/cgi-bin/hgTracks?db=hg38&lastVirtModeType=default&lastVirtModeExtraState=&virtModeType=default&virtMode=0&nonVirtPosition=&position=chr22%3A45950143%2D45950148&hgsid=572048045_LXaRz5ejmC9V6zso2TTWMLapbn6a
$inputAref = [['chr1', 115286071, 'C', 'T', 'SNP', 'C', 1, 'T', 1]];

$outAref = $annotator->addTrackData($inputAref);

@outData = @{$outAref->[0]};

$geneTrackData = $outData[$trackIndices->{refSeq}];

# TODO: maybe export the types of names from the gene track package
ok($outData[4][0][0] eq 'T', 'The alt allele is an T');
ok($geneTrackData->[$exonicAlleleFunctionIdx][0][0] eq 'synonymous', 'We agree with UCSC that chr1:45950143 C>T is synonymous (stop -> stop)');
ok($geneTrackData->[$refAAidx][0][0] eq '*', 'We agree with UCSC that chr1:45950143 is a stop');
ok($geneTrackData->[$altAAidx][0][0] eq '*', 'We agree with UCSC that chr1:45950143 is a stop');

# Note that the CDS end in refGene is 115286795
my $pos = 115286796;
say "\n\nTesting a UTR5 site SNP on the negative strand gene NGF\n";

$inputAref = [['chr1', $pos, 'T', 'C,G', 'SNP', 'C', 1, 'G', 1]];

$outAref = $annotator->addTrackData($inputAref);

@outData = @{$outAref->[0]};

$geneTrackData = $outData[$trackIndices->{refSeq}];

$geneTrackRegionData = $db->dbReadAll( $geneTrack->regionTrackPath('chr1') );

# p $geneTrackRegionData->{3755};
# say "and 5";
# p $geneTrackRegionData->{5};

$sth = $dbh->prepare("SELECT * FROM refGene WHERE refGene.name = 'NM_002506'");
$sth->execute();
@row = $sth->fetchrow_array;

my $cdsEnd = $row[7];

ok($outData[4][0][0] eq 'C', 'The alt allele is a G');
# TODO: maybe export the types of names from the gene track package
# cdsEnd is weirdly closed, despite the rest of refGene being half-open....
if($pos >= $cdsEnd) {
  ok($geneTrackData->[$siteTypeIdx][0][0] eq 'UTR5', 'We agree with UCSC that chr1:115286796 is in the UTR5 of NGF');
} else {
  ok($geneTrackData->[$siteTypeIdx][0][0] eq 'exonic', 'We agree with UCSC that chr1:115286796 is in the last exon of NGF');
}

ok(!defined $geneTrackData->[$exonicAlleleFunctionIdx][0][0], 'UTR5 sites don\'t have exonicAlleleFunction');
ok(!defined $geneTrackData->[$refAAidx][0][0], 'UTR5 sites don\'t have reference amino acids');
ok(!defined $geneTrackData->[$altAAidx][0][0], 'UTR5 sites don\'t have allele amino acids');
ok($geneTrackData->[$geneSymbolIdx][0][0] eq 'NGF', 'We agree with UCSC that chr1:115286069 geneSymbol is NGF');

# NGF is weird, here is the refGene table:
# mysql> SELECT * FROM refGene WHERE refGene.name2 = 'NGF';
# +------+-----------+-------+--------+-----------+-----------+-----------+-----------+-----------+--------------------------------+--------------------------------+-------+-------+--------------+------------+------------+
# | bin  | name      | chrom | strand | txStart   | txEnd     | cdsStart  | cdsEnd    | exonCount | exonStarts                     | exonEnds                       | score | name2 | cdsStartStat | cdsEndStat | exonFrames |
# +------+-----------+-------+--------+-----------+-----------+-----------+-----------+-----------+--------------------------------+--------------------------------+-------+-------+--------------+------------+------------+
# | 1464 | NM_002506 | chr1  | -      | 115285915 | 115338236 | 115286069 | 115286795 |         3 | 115285915,115293626,115338203, | 115286807,115293750,115338236, |     0 | NGF   | cmpl         | cmpl       | 0,-1,-1,   |
# +------+-----------+-------+--------+-----------+-----------+-----------+-----------+-----------+--------------------------------+--------------------------------+-------+-------+--------------+------------+------------+

# It looks like all of the exonEnds are past the cdsEnd, so ther cannot be a spliceDonor/Acceptor
# my @exonEnds =  split(',', $row[10]);

# my $firstExonHalfClosed = $exonEnds[1];

# p $firstExonHalfClosed;


# say "\n\nTesting a UTR5 site SNP on the negative strand gene NGF\n";


# $inputAref = [['chr22', $firstExonHalfClosed, 'T', 'C', 'SNP', 'C', 1, 'G', 1]];
# ($annotator->{_genoNames}, $annotator->{_genosIdx}, $annotator->{_confIdx}) = 
#   (["Sample_4", "Sample_5"], [5, 7], [6, 8]);

# $annotator->{_genosIdxRange} = [0, 1];

# $dataAref = $db->dbRead('chr1', [$firstExonHalfClosed - 1]);

# p $dataAref;

# $outAref = [];

# $annotator->addTrackData('chr1', $dataAref, $inputAref, $outAref);

# @outData = @{$outAref->[0]};

# $geneTrackData = $outData[$trackIndices->{refSeq}];
# p $geneTrackData->[$siteTypeIdx][0][0];
# ok($geneTrackData->[$siteTypeIdx][0][0] eq 'spliceAcceptor', "We agree with UCSC that chr1\:$firstExonHalfClosed is in the last exon of NGF");
# ok(!defined $geneTrackData->[$exonicAlleleFunctionIdx][0][0], 'splice sites don\'t have exonicAlleleFunction');
# ok(!defined $geneTrackData->[$refAAidx][0][0], 'splice sites don\'t have reference amino acids');
# ok(!defined $geneTrackData->[$altAAidx][0][0], 'splice sites don\'t have allele amino acids');
# ok($geneTrackData->[$geneSymbolIdx][0][0] eq 'NGF', 'We agree with UCSC that chr1:115286069 geneSymbol is NGF');

# TODO: Test spliceDonor/spliceAcceptor

say "\nTesting chr1:115293629 which should be UTR5\n";

$inputAref = [['chr1', 115293629, 'T', 'C,G', 'SNP', 'C', 1, 'G', 1]];

$outAref = $annotator->addTrackData($inputAref);

@outData = @{$outAref->[0]};

$geneTrackData = $outData[$trackIndices->{refSeq}];

ok($geneTrackData->[$siteTypeIdx][0][0] eq 'UTR5', "chr1\:115293629 is in the UTR5 NGF");
ok(!defined $geneTrackData->[$exonicAlleleFunctionIdx][0][0], 'UTR5 sites don\'t have exonicAlleleFunction');
ok(!defined $geneTrackData->[$refAAidx][0][0], 'UTR5 sites don\'t have reference amino acids');
ok(!defined $geneTrackData->[$altAAidx][0][0], 'UTR5 sites don\'t have allele amino acids');
ok($geneTrackData->[$geneSymbolIdx][0][0] eq 'NGF', 'We agree with UCSC that chr1:115286069 geneSymbol is NGF');

# TODO: Test spliceDonor/spliceAccptor

say "\nTesting chr1:115338204 which should be UTR5\n";

$inputAref = [['chr1', 115338204, 'T', 'C,G', 'SNP', 'C', 1, 'G', 1]];

$outAref = $annotator->addTrackData($inputAref);

@outData = @{$outAref->[0]};

$geneTrackData = $outData[$trackIndices->{refSeq}];

ok($geneTrackData->[$siteTypeIdx][0][0] eq 'UTR5', "chr1\:115338204 is in the UTR5 of NGF");
ok(!defined $geneTrackData->[$exonicAlleleFunctionIdx][0][0], 'UTR5 sites don\'t have exonicAlleleFunction');
ok(!defined $geneTrackData->[$refAAidx][0][0], 'UTR5 sites don\'t have reference amino acids');
ok(!defined $geneTrackData->[$altAAidx][0][0], 'UTR5 sites don\'t have allele amino acids');
ok($geneTrackData->[$geneSymbolIdx][0][0] eq 'NGF', 'We agree with UCSC that chr1:115286069 geneSymbol is NGF');


say "\nTesting chr1:115286806 for splice\n";

$inputAref = [['chr1', 115286806, 'T', 'C,G', 'SNP', 'C', 1, 'G', 1]];

$outAref = $annotator->addTrackData($inputAref);

@outData = @{$outAref->[0]};

$geneTrackData = $outData[$trackIndices->{refSeq}];
p $geneTrackData;
ok($geneTrackData->[$siteTypeIdx][0][0] eq 'UTR5', "We agree with UCSC that chr1\:115286806 is in the UTR of NGF");
ok(!defined $geneTrackData->[$exonicAlleleFunctionIdx][0][0], 'splice sites don\'t have exonicAlleleFunction');
ok(!defined $geneTrackData->[$refAAidx][0][0], 'splice sites don\'t have reference amino acids');
ok(!defined $geneTrackData->[$altAAidx][0][0], 'splice sites don\'t have allele amino acids');
ok($geneTrackData->[$geneSymbolIdx][0][0] eq 'NGF', 'We agree with UCSC that chr1:115286069 geneSymbol is NGF');


# TEST PhastCons && PhyloP
use Seq::Tracks::Score::Build::Round;

my $rounder = Seq::Tracks::Score::Build::Round->new();

my $refTrackGetter = $tracks->getRefTrackGetter();
# say "\nTesting PhastCons 100 way\n";



# $inputAref = [['chr22', 24772303, 'T', 'C', 'SNP', 'C', 1, 'G', 1]];
# ($annotator->{_genoNames}, $annotator->{_genosIdx}, $annotator->{_confIdx}) = 
#   (["Sample_4", "Sample_5"], [5, 7], [6, 8]);

# $annotator->{_genosIdxRange} = [0, 1];

# $dataAref = $db->dbRead('chr22', [24772303 - 1]);

# p $dataAref;

# $outAref = [];

# $annotator->addTrackData('chr22', $dataAref, $inputAref, $outAref);

# @outData = @{$outAref->[0]};

# my $phastConsData = $outData[$trackIndices->{phastCons}];

# p $phastConsData;

# ok($phastConsData == $rounder->round(.802) + 0, "chr22\:24772303 has a phastCons score of .802 (rounded)");

# say "\nTesting chr1:115286806 for splice\n";

# $inputAref = [['chr22', 115286806, 'T', 'C', 'SNP', 'C', 1, 'G', 1]];
# ($annotator->{_genoNames}, $annotator->{_genosIdx}, $annotator->{_confIdx}) = 
#   (["Sample_4", "Sample_5"], [5, 7], [6, 8]);

# $annotator->{_genosIdxRange} = [0, 1];

# $dataAref = $db->dbRead('chr1', [115286806 - 1]);

# p $dataAref;

# $outAref = [];

# $annotator->addTrackData('chr1', $dataAref, $inputAref, $outAref);

# @outData = @{$outAref->[0]};

# $geneTrackData = $outData[$trackIndices->{refSeq}];

# p $geneTrackData;

# p $geneTrackData->[$siteTypeIdx][0][0];
# ok($geneTrackData->[$siteTypeIdx][0][0] eq 'spliceAcceptor', "We agree with UCSC that chr1\:115286806 is in the last exon of NGF");

# Testing PhyloP

say "\nTesting PhyloP 100 way chr22:24772200-24772303 db data\n";

#1-start coordinate system in use for variableStep and fixedStep

# bigWigs that represent variableStep and fixedStep data are generated from wiggle files which use "1-start, fully-closed" coordinates. For example, for a chromosome of length N, the first position is 1 and the last position is N. For more information, see this related article: BigWig and BigBed: enabling browsing of large distributed datasets, this FAQ, or this blog.
# UCSC NUMBERS DON'T MATCH THEIR OWN GOLDENPATH DOWNLOADED FILE NUMBERS
# Table browser
# track type=wiggle_0 name="Cons 100 Verts" description="100 vertebrates Basewise Conservation by PhyloP"
# output date: 2017-02-03 21:02:37 UTC
# chrom specified: chr22
# position specified: 10510000-10510022
# This data has been compressed with a minor loss in resolution.
# (Worst case: 0.0418359)  The original source data
# (before querying and compression) is available at 
#   http://hgdownload.cse.ucsc.edu/downloads.html
# variableStep chrom=chr22 span=1
# 10510001  1.08368
# 10510002  1.04151
# 10510003  1.12584
# 10510004  0.113874
# 10510005  -0.265614
# 10510006  -1.57274
# 10510007  0.113874
# 10510008  0.113874
# 10510009  -0.139118
# 10510010  1.08368
# 10510011  -0.434276
# 10510012  1.08368
# 10510013  0.156039
# 10510014  0.113874
# 10510015  1.21017
# 10510016  -0.223449
# 10510017  -1.15109
# 10510018  1.08368
# 10510019  -0.645102
# 10510020  1.21017
# 10510021  0.156039
# 10510022  1.21017

#Goldenpath (using this)
# fixedStep chrom=chr22 start=10510001 step=1
# 1.123
# 1.059
# 1.163
# 0.117
# -0.243
# -1.561
# 0.117
# 0.151
# -0.116
# 1.084
# -0.429
# 1.084
# 0.161
# 0.156
# 1.246
# -0.202
# -1.124
# 1.084
# -0.636
# 1.246
# 0.157
# 1.246
# -0.200
# -0.097
# -0.051
# 1.084
# -1.024
# 0.149
# -0.199
# -0.216
# 1.084
# -0.079
# -0.648
# -0.154
# -1.288
# -0.139
# 1.084
# 0.104
# 0.113
# 1.084
# -0.576
# 0.031
# 0.893
# 0.893
# 0.147
# 0.222
# -0.228
# 0.225
# -0.766
# 1.163
# -0.256
# 1.163
# 1.163
# 0.212
# 1.246
# -0.716
# -0.293
# -0.171
# -0.556
# -1.338
# 0.275
# 0.243
# -0.253
# -1.558
# 1.202
# 0.250
# 0.135
# 0.250
# -0.263
# -0.551
# 0.257
# 0.251


############### The test ####################
my @ucscPhyloP100way = (1.123,1.059,1.163,0.117,-0.243,-1.561,0.117,0.151,-0.116,1.084,-0.429,1.084,0.161,0.156,1.246,-0.202,-1.124,1.084,-0.636,1.246,0.157,1.246,-0.200,-0.097,-0.051,1.084,-1.024,0.149,-0.199,-0.216,1.084,-0.079,-0.648,-0.154,-1.288,-0.139,1.084,0.104,0.113,1.084,-0.576,0.031,0.893,0.893,0.147,0.222,-0.228,0.225,-0.766,1.163,-0.256,1.163,1.163,0.212,1.246,-0.716,-0.293,-0.171,-0.556,-1.338,0.275,0.243,-0.253,-1.558,1.202,0.250,0.135,0.250,-0.263,-0.551,0.257,0.251);
my @positions = ( 10510001 - 1 .. (10510001 - 1 + @ucscPhyloP100way) - 1 );

my @dbData = @positions;
$db->dbRead('chr22', \@dbData);

my $phyloPTrack = $tracks->getTrackGetterByName('phyloP');

my $i = 0;
for my $data (@dbData) {
  my $phyloPscore = $data->[$phyloPTrack->dbName];

  my $ucscRounded = $rounder->round($ucscPhyloP100way[$i]) + 0;

  # say "ours phastCons: $phastConsData theirs: " . $rounder->round($ucscPhyloP100way[$i]);
  ok($phyloPscore == $ucscRounded, "chr22\:$positions[$i]: our phyloP score: $phyloPscore ; theirs: $ucscRounded ; exact: $ucscPhyloP100way[$i]");
  $i++;
}

say "\nTesting PhyloP 100 way chr22:24772200-24772303 output data\n";

$inputAref = [];
my @alleles = ('A', 'C', 'T', 'G');
for my $i ( 0 .. $#positions) {
  my $refBase = $refTrackGetter->get( $dbData[$i] );

  my $nonRef = first { $_ ne $refBase}@alleles;

  $inputAref->[$i] = ['chr22', $positions[$i] + 1, $refBase, $nonRef, 'SNP', $refBase, 1, $nonRef, 1]
}

$outAref = $annotator->addTrackData($inputAref);

$i = 0;
for my $data (@$outAref) {
  my $phyloPscore = $data->[$trackIndices->{phyloP}][0][0];

  my $ucscRounded = $rounder->round($ucscPhyloP100way[$i]) + 0;

  # say "ours phastCons: $phastConsData theirs: " . $rounder->round($ucscPhyloP100way[$i]);
  ok($phyloPscore == $ucscRounded, "chr22\:$positions[$i]: our phyloP score: $phyloPscore ; theirs: $ucscRounded ; exact: $ucscPhyloP100way[$i]");
  $i++;
}

say "\nTesting PhyloP 7 way chr22:50800878-50800977\n";
# --
# fixedStep chrom=chr22 start=50800878 step=1
# 0.097
# 0.108
# 0.112
# 0.108
# 0.096
# -1.948
# 0.106
# 0.105
# 0.106
# 0.118
# 0.105
# 0.106
# 0.106
# 0.123
# 0.118
# -1.602
# 0.137
# 0.160
# 0.160
# 0.153
# 0.137
# 0.137
# -1.415
# 0.153
# -1.338
# 0.153
# 0.160
# 0.153
# 0.136
# 0.153
# 0.136
# 0.153
# 0.136
# 0.153
# -1.392
# 0.153
# 0.160
# 0.136
# 0.137
# 0.160
# 0.160
# 0.137
# -1.609
# 0.153
# 0.153
# 0.153
# 0.136
# 0.153
# 0.136
# -2.891
# 0.153
# 0.137
# 0.137
# 0.137
# 0.136
# 0.137
# 0.160
# 0.160
# 0.160
# 0.136
# 0.153
# 0.137
# 0.153
# 0.137
# 0.136
# 0.137
# -1.290
# 0.137
# 0.160
# 0.153
# -1.288
# 0.160
# -1.236
# 0.137
# 0.160
# 0.137
# 0.137
# 0.137
# 0.136
# 0.137
# 0.160
# 0.160
# 0.153
# 0.153
# 0.160
# 0.153
# 0.136
# 0.153
# 0.136
# 0.137
# 0.160
# 0.153
# 0.160
# 0.153
# 0.137
# 0.153
# 0.153
# 0.136
# 0.136
# -1.405

@ucscPhyloP100way = (0.097,0.108,0.112,0.108,0.096,-1.948,0.106,0.105,0.106,0.118,0.105,0.106,0.106,0.123,0.118,-1.602,0.137,0.160,0.160,0.153,0.137,0.137,-1.415,0.153,-1.338,0.153,0.160,0.153,0.136,0.153,0.136,0.153,0.136,0.153,-1.392,0.153,0.160,0.136,0.137,0.160,0.160,0.137,-1.609,0.153,0.153,0.153,0.136,0.153,0.136,-2.891,0.153,0.137,0.137,0.137,0.136,0.137,0.160,0.160,0.160,0.136,0.153,0.137,0.153,0.137,0.136,0.137,-1.290,0.137,0.160,0.153,-1.288,0.160,-1.236,0.137,0.160,0.137,0.137,0.137,0.136,0.137,0.160,0.160,0.153,0.153,0.160,0.153,0.136,0.153,0.136,0.137,0.160,0.153,0.160,0.153,0.137,0.153,0.153,0.136,0.136,-1.405);
@positions = (50800878 .. 50800878 - 1 + @ucscPhyloP100way);

@dbData = @positions;
$dbData[0] = $dbData[0] - 1;
$dbData[-1] = $dbData[-1] - 1;
$db->dbRead('chr22', \@dbData);

# my $phyloPTrack = $tracks->getTrackGetterByName('phyloP');

# my $i = 0;
# for my $data (@dbData) {
#   my $phyloPscore = $data->[$phyloPTrack->dbName];

#   my $ucscRounded = $rounder->round($ucscPhyloP100way[$i]) + 0;

#   # say "ours phastCons: $phastConsData theirs: " . $rounder->round($ucscPhyloP100way[$i]);
#   ok($phyloPscore == $ucscRounded, "chr22\:$positions[$i]: our phyloP score: $phyloPscore ; theirs: $ucscRounded ; exact: $ucscPhyloP100way[$i]");
#   $i++;
# }

# say "\nTesting PhyloP 100 way chr22:24772200-24772303 output data\n";

$inputAref = [];
@alleles = ('A', 'C', 'T', 'G');
for my $i ( 0 .. $#positions) {
  my $refBase = $refTrackGetter->get( $dbData[$i] );

  my $nonRef = first { $_ ne $refBase}@alleles;

  $inputAref->[$i] = ['chr22', $positions[$i], $refBase, $nonRef, 'SNP', $refBase, 1, $nonRef, 1]
}

$outAref = $annotator->addTrackData($inputAref);

$i = 0;
for my $data (@$outAref) {
  my $phyloPscore = $data->[$trackIndices->{phyloP}][0][0];

  my $ucscRounded = $rounder->round($ucscPhyloP100way[$i]) + 0;

  # say "ours phastCons: $phastConsData theirs: " . $rounder->round($ucscPhyloP100way[$i]);
  ok($phyloPscore == $ucscRounded, "chr22\:$positions[$i]: our phyloP score: $phyloPscore ; theirs: $ucscRounded ; exact: $ucscPhyloP100way[$i]");
  $i++;
}


say "\nTesting 100 phyloP100way sites from fixedStep chrom=chr22 start=38356615 step=1\n";
# --
# fixedStep chrom=chr22 start=38356615 step=1
# 0.055
# 0.055
# 0.089
# 0.085
# 0.076
# 0.076
# 0.077
# 0.076
# 0.085
# 0.077
# 0.076
# 0.077
# 0.077
# 0.076
# 0.085
# 0.077
# 0.089
# 0.077
# 0.077
# 0.089
# 0.077
# 0.085
# 0.077
# 0.085
# 0.076
# 0.076
# 0.077
# 0.089
# 0.076
# 0.085
# 0.077
# 0.077
# 0.077
# 0.077
# 0.076
# 0.085
# 0.076
# 0.077
# 0.089
# 0.076
# 0.085
# 0.085
# 0.085
# 0.076
# 0.076
# 0.077
# 0.089
# 0.077
# 0.076
# 0.085
# 0.076
# 0.085
# 0.089
# 0.085
# 0.085
# 0.089
# 0.089
# 0.077
# 0.076
# 0.089
# 0.089
# 0.089
# 0.077
# -1.893
# 0.077
# 0.089
# 0.089
# 0.085
# 0.085
# 0.077
# 0.077
# 0.077
# 0.077
# 0.076
# 0.076
# 0.085
# 0.076
# 0.077
# 0.077
# 0.077
# 0.077
# 0.089
# 0.089
# 0.076
# 0.085
# 0.077
# 0.076
# 0.085
# 0.077
# 0.089
# 0.077
# 0.077
# 0.089
# 0.089
# 0.076
# 0.085
# 0.076
# 0.077
# 0.076
# 0.085
# --

@ucscPhyloP100way = (0.055,0.055,0.089,0.085,0.076,0.076,0.077,0.076,0.085,0.077,0.076,0.077,0.077,0.076,0.085,0.077,0.089,0.077,0.077,0.089,0.077,0.085,0.077,0.085,0.076,0.076,0.077,0.089,0.076,0.085,0.077,0.077,0.077,0.077,0.076,0.085,0.076,0.077,0.089,0.076,0.085,0.085,0.085,0.076,0.076,0.077,0.089,0.077,0.076,0.085,0.076,0.085,0.089,0.085,0.085,0.089,0.089,0.077,0.076,0.089,0.089,0.089,0.077,-1.893,0.077,0.089,0.089,0.085,0.085,0.077,0.077,0.077,0.077,0.076,0.076,0.085,0.076,0.077,0.077,0.077,0.077,0.089,0.089,0.076,0.085,0.077,0.076,0.085,0.077,0.089,0.077,0.077,0.089,0.089,0.076,0.085,0.076,0.077,0.076,0.085);
@positions = (38356615  .. 38356615 + @ucscPhyloP100way - 1);

@dbData = @positions;
$dbData[0] = $dbData[0] - 1;
$dbData[-1] = $dbData[-1] - 1;
$db->dbRead('chr22', \@dbData);

$inputAref = [];
@alleles = ('A', 'C', 'T', 'G');
for my $i ( 0 .. $#positions) {
  my $refBase = $refTrackGetter->get( $dbData[$i] );

  my $nonRef = first { $_ ne $refBase}@alleles;

  $inputAref->[$i] = ['chr22', $positions[$i], $refBase, $nonRef, 'SNP', $refBase, 1, $nonRef, 1]
}

$outAref = $annotator->addTrackData($inputAref);

$i = 0;
for my $data (@$outAref) {
  my $phyloPscore = $data->[$trackIndices->{phyloP}][0][0];

  my $ucscRounded = $rounder->round($ucscPhyloP100way[$i]) + 0;

  # say "ours phastCons: $phastConsData theirs: " . $rounder->round($ucscPhyloP100way[$i]);
  ok($phyloPscore == $ucscRounded, "chr22\:$positions[$i]: our phyloP score: $phyloPscore ; theirs: $ucscRounded ; exact: $ucscPhyloP100way[$i]");
  $i++;
}



say "\nTesting 100 phyloP100way sites from fixedStep chrom=chr22 start=21299133 step=1\n";
# --
# fixedStep chrom=chr22 start=21299133 step=1
# 0.155
# 0.138
# 0.139
# 0.138
# 0.162
# 0.162
# 0.162
# 0.155
# 0.162
# -1.261
# 0.138
# 0.139
# 0.162
# 0.155
# 0.162
# 0.162
# 0.155
# 0.162
# 0.162
# 0.162
# 0.155
# 0.162
# 0.155
# 0.155
# 0.139
# 0.138
# -1.261
# 0.139
# 0.139
# 0.139
# 0.162
# 0.155
# 0.155
# 0.138
# 0.138
# 0.139
# -1.261
# 0.162
# 0.162
# 0.155
# 0.162
# 0.162
# 0.138
# 0.162
# 0.162
# 0.155
# 0.162
# 0.162
# 0.139
# 0.139
# 0.162
# 0.138
# 0.155
# -1.682
# 0.139
# 0.162
# 0.155
# 0.162
# 0.138
# 0.138
# 0.155
# 0.155
# 0.162
# 0.155
# 0.139
# 0.139
# 0.162
# -1.432
# 0.162
# 0.138
# 0.138
# 0.155
# 0.138
# 0.139
# 0.162
# 0.138
# 0.155
# 0.138
# 0.139
# 0.138
# -1.432
# 0.155
# 0.162
# 0.138
# 0.138
# 0.139
# 0.162
# 0.162
# 0.162
# 0.138
# -1.676
# 0.155
# 0.138
# 0.155
# 0.162
# 0.162
# 0.162
# 0.138
# -1.798
# 0.155
# --

@ucscPhyloP100way = (0.155,0.138,0.139,0.138,0.162,0.162,0.162,0.155,0.162,-1.261,0.138,0.139,0.162,0.155,0.162,0.162,0.155,0.162,0.162,0.162,0.155,0.162,0.155,0.155,0.139,0.138,-1.261,0.139,0.139,0.139,0.162,0.155,0.155,0.138,0.138,0.139,-1.261,0.162,0.162,0.155,0.162,0.162,0.138,0.162,0.162,0.155,0.162,0.162,0.139,0.139,0.162,0.138,0.155,-1.682,0.139,0.162,0.155,0.162,0.138,0.138,0.155,0.155,0.162,0.155,0.139,0.139,0.162,-1.432,0.162,0.138,0.138,0.155,0.138,0.139,0.162,0.138,0.155,0.138,0.139,0.138,-1.432,0.155,0.162,0.138,0.138,0.139,0.162,0.162,0.162,0.138,-1.676,0.155,0.138,0.155,0.162,0.162,0.162,0.138,-1.798,0.155);
@positions = (21299133 .. 21299133 + @ucscPhyloP100way - 1);

@dbData = @positions;
$dbData[0] = $dbData[0] - 1;
$dbData[-1] = $dbData[-1] - 1;
$db->dbRead('chr22', \@dbData);

$inputAref = [];
@alleles = ('A', 'C', 'T', 'G');
for my $i ( 0 .. $#positions) {
  my $refBase = $refTrackGetter->get( $dbData[$i] );

  my $nonRef = first { $_ ne $refBase}@alleles;

  $inputAref->[$i] = ['chr22', $positions[$i], $refBase, $nonRef, 'SNP', $refBase, 1, $nonRef, 1]
}

$outAref = $annotator->addTrackData($inputAref);

$i = 0;
for my $data (@$outAref) {
  my $phyloPscore = $data->[$trackIndices->{phyloP}][0][0];

  my $ucscRounded = $rounder->round($ucscPhyloP100way[$i]) + 0;

  # say "ours phastCons: $phastConsData theirs: " . $rounder->round($ucscPhyloP100way[$i]);
  ok($phyloPscore == $ucscRounded, "chr22\:$positions[$i]: our phyloP score: $phyloPscore ; theirs: $ucscRounded ; exact: $ucscPhyloP100way[$i]");
  $i++;
}

#dbSNP
# bed format is start (0-based) , end (open, aka subtract one to get real end)
# 1349  chr1  100140410 100140422 rs868129983 0 + TGAATTCTACAC  TGAATTCTACAC  -/TGAATTCTACAC  genomic deletion  unknown 0 0 intron,ncRNA,cds-indel  range 1 NA  1 ILLUMINA, 0 NA  NA  NA  NA

# these are for input file, so 1-based
@positions = (100140410 + 1 .. 100140422 + 10);

# make zero-based
@dbData = @positions;
$dbData[0] = $dbData[0] - 1;
$dbData[-1] = $dbData[-1] - 1;
$db->dbRead('chr1', \@dbData);

my $dbSnpTrack = $tracks->getTrackGetterByName('dbSNP');
# my $dbSNPnameDbName = $dbSnpTrack->getFieldDbName('name');

$inputAref = [];
@alleles = ('A', 'C', 'T', 'G');
for my $i ( 0 .. $#positions) {
  my $refBase = $refTrackGetter->get( $dbData[$i] );

  my $nonRef = first { $_ ne $refBase}@alleles;

  $inputAref->[$i] = ['chr1', $positions[$i], $refBase, $nonRef, 'SNP', $refBase, 1, $nonRef, 1]
}

$outAref = $annotator->addTrackData($inputAref);

$i = 0;
for my $data (@$outAref) {
  my $rsNumber = $data->[$trackIndices->{dbSNP}][0][0];
  if($data->[1] > 100140422) {
    ok(first_index { $_ eq 'rs868129983' } @$rsNumber == -1, "chr1\:$positions[$i] (a base past rs868129983 doesn't have rs868129983");
    $i++;
    next;
  }
  
  ok(first_index { $_ eq 'rs868129983' } @$rsNumber > -1, "chr1\:$positions[$i] has rs868129983");
  $i++;
}

system('rm test.snp');
done_testing();
