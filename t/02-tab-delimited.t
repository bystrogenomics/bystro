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
use Seq::Headers;
use List::Util qw/first all none/;
use List::MoreUtils qw/first_index/;

system('touch test.snp && echo "Fragment,Position,Reference,Alleles,Allele_Counts,Type,SL106190,SL106191" > $_');

my $annotator = MockAnnotationClass->new_with_config({ config => './config/hg38.yml', verbose => 1});

my ($err, $stats, $outFiles) = $annotator->annotate();

ok($err eq "Bystro accepts only tab delimited files. Please reformat your file", "Bystro requires tab delimited files");

system("echo 'Fragment\tPosition\tReference Alleles\tAllele_Counts\tType\tSL106190\tSL106191' > test.snp");

$annotator = MockAnnotationClass->new_with_config({ config => './config/hg38.yml', verbose => 1});

($err, $stats, $outFiles) = $annotator->annotate();

ok(!defined $err, "Bystro annotates tab delimited files, even if they are wholly empty besides header");
# system('rm test.snp');
done_testing();