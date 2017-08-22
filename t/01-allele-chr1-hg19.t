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
system('touch test.snp');

my $annotator = MockAnnotationClass->new_with_config({ config => './config/hg19.yml'});

system('rm test.snp');

my $sqlClient = Utils::SqlWriter::Connection->new();

my $dbh = $sqlClient->connect('hg19');

my $tracks = Seq::Tracks->new({tracks => $annotator->tracks, gettersOnly => 1});

my $db = Seq::DBManager->new();
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
# my $sth = $dbh->prepare('SELECT * FROM hg38.refGene WHERE chrom="chr22" AND (txStart <= 14000001 AND txEnd>=14000001) OR (txStart >= 14000001 AND txEnd<=14000001)');

# $sth->execute();

# #Schema:
# #   0 , 1   , 2    , 3      ... 12
# # [bin, name, chrom, strand, ...name2,
# # 
# my @row = $sth->fetchrow_array;

# ok(!@row, "UCSC still has 14000001 as intergenic");
# # ok($geneTrackData->[0][0][0] eq 'intergenic', "We have this as intergenic");


# Check a site with 1 transcript, on the negative strand



$geneTrackData = $outData[$trackIndices->{refSeq}];

# $sth = $dbh->prepare("SELECT * FROM hg38.refGene WHERE chrom='chr22' AND (txStart <= 45950000 AND txEnd>=45950000) OR (txStart >= 45950000 AND txEnd<=45950000);");

# $sth->execute();

# @row = $sth->fetchrow_array;

# TODO: move away from requiring ref name.
my $refTrackIdx = $trackIndices->{ref};
my $txNameIdx = $geneTrackFeatureMap{refseq};
my $geneSymbolIdx = $geneTrackFeatureMap{geneSymbol};
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

### Weird site
# http://genome.ucsc.edu/cgi-bin/hgc?hgsid=590836883_egoiRESddM5GTAEeLusrOmdT1Daz&c=chr1&l=67672592&r=67672593&o=67632168&t=67725650&g=refGene&i=NM_144701
# In exon, but it doesn't seem like we have associated info for it

say "\nTesting chr1:67,672,592-593 which should be ...?\n";

$inputAref = [['chr1', 67672592, 'T', '-2', 'DEL', 'D', 1]];

$outAref = $annotator->addTrackData($inputAref);

@outData = @{$outAref->[0]};

$geneTrackData = $outData[$trackIndices->{refSeq}];

p $outAref;
p $geneTrackData;

say "\nTesting chr1:67,672,592-593 insertion which should be ...?\n";

$inputAref = [['chr1', 67672592, 'T', '+TTT', 'INS', 'I', 1]];

$outAref = $annotator->addTrackData($inputAref);

@outData = @{$outAref->[0]};

$geneTrackData = $outData[$trackIndices->{refSeq}];

p $outAref;
p $geneTrackData;
exit;
done_testing();
