use 5.10.0;
use strict;
use warnings;

package Seq::Definition;
use Mouse::Role 2;
use Path::Tiny;
use DDP;
use Types::Path::Tiny qw/AbsPath AbsFile AbsDir/;
use List::MoreUtils qw/first_index/;
use Mouse::Util::TypeConstraints;

use Seq::Tracks;
use Seq::Statistics;

with 'Seq::Role::IO';
with 'Seq::Output::Fields';
# Note: All init_arg undef methods must be lazy if they rely on arguments that are
# not init_arg => undef, and do not have defaults (aka are required)
######################## Required ##############################

# output_file_base contains the absolute path to a file base name
# Ex: /dir/child/BaseName ; BaseName is appended with .annotated.tsv , .annotated-log.txt, etc
# for the various outputs
######################## Required ##############################

# output_file_base contains the absolute path to a file base name
# Ex: /dir/child/BaseName ; BaseName is appended with .annotated.tsv , .annotated-log.txt, etc
# for the various outputs
has output_file_base => ( is => 'ro', isa => AbsPath, coerce => 1, required => 1,
  handles => { outDir => 'parent', outBaseName => 'basename' });

############################### Optional #####################################
# String, allowing us to ignore it if not truthy
# Acceptable values include ~ in YAML (undef/null)
has temp_dir => (is => 'ro', isa => 'Maybe[Str]');

# Do we want to compress?
has compress => (is => 'ro', isa => 'Bool', default => 1);

# Do we want to tarball our results
has archive => (is => 'ro', isa => 'Bool', default => 0);

# The statistics configuration options, usually defined in a YAML config file
has statistics => (is => 'ro', isa => 'HashRef');

# Users may not need statistics
has run_statistics => (is => 'ro', isa => 'Bool', default => sub {!!$_[0]->statistics});

has maxThreads => (is => 'ro', isa => 'Int', lazy => 1, default => 8);

# has badSamplesField => (is => 'ro', default => 'badSamples', lazy => 1);

################ Public Exports ##################
#@ params
# <Object> filePaths @params:
  # <String> compressed : the name of the compressed folder holding annotation, stats, etc (only if $self->compress)
  # <String> converted : the name of the converted folder
  # <String> annnotation : the name of the annotation file
  # <String> log : the name of the log file
  # <Object> stats : the { statType => statFileName } object
# Allows us to use all to to extract just the file we're interested from the compressed tarball
has outputFilesInfo => (is => 'ro', isa => 'HashRef', init_arg => undef, lazy => 1, default => sub {
  my $self = shift;

  my %out;

  $out{log} = path($self->logPath)->basename;

  # Must be lazy in order to allow "revealing module pattern", with output_file_base below
  my $outBaseName = $self->outBaseName;

  $out{annotation} = $outBaseName . '.annotation.tsv' . ($self->compress ? ".gz" : "");

  # Must be lazy in order to allow "revealing module pattern", with __statisticsRunner below
  if($self->run_statistics) {
    $out{statistics} = {
      json => path($self->_statisticsRunner->jsonFilePath)->basename,
      tab => path($self->_statisticsRunner->tabFilePath)->basename,
      qc => path($self->_statisticsRunner->qcFilePath)->basename,
    };
  }

  if($self->archive) {
    # Seq::Role::IO method
    # Only compress the tarball if we're not compressing the inner file
    # because this wastes a lot of time, since the compressed inner annotation
    # which dominates the archive 99%, cannot be compressed at all
    $out{archived} = $self->makeTarballName($outBaseName, !$self->compress);
  }

  return \%out;
});

############################ Private ###################################
# Must be lazy... Mouse doesn't seem to respect attribute definition order at all times
# Leading to situations where $self->outDir doesn't exist by the time _workingDir
# is created. This can lead to the contents of the current working directory being accidentally compressed
# into $self->outputFilesInfo->{archived}
has _workingDir => (is => 'ro', init_arg => undef, lazy => 1, default => sub {
  my $self = shift;


  if($self->temp_dir) {
    my $dir = path($self->temp_dir);
    $dir->mkpath;

    return Path::Tiny->tempdir(DIR => $dir, CLEANUP => 1)
  }

  return $self->outDir;
});

### Override logPath to use the working directory / output_file_base basename ###
has logPath => (is => 'ro', init_arg => undef, lazy => 1, default => sub {
  my $self = shift;
  return $self->_workingDir->child($self->outBaseName. '.annotation.log.txt')->stringify();
});

# Must be lazy because needs run_statistics and statistics
has _statisticsRunner => (is => 'ro', init_arg => undef, lazy => 1, default => sub {
  my $self = shift;

  my $basePath = $self->_workingDir->child($self->outBaseName)->stringify;
  # Assumes that is run_statistics is specified, $self-statistics exists
  if($self->run_statistics) {
    my %args = (
      altField => $self->altField,
      homozygotesField => $self->homozygotesField,
      heterozygotesField => $self->heterozygotesField,
      outputBasePath => $basePath,
    );

    %args = (%args, %{$self->statistics});

    return Seq::Statistics->new(\%args);
  }

  return undef;
});

sub _moveFilesToOutputDir {
  my $self = shift;

  my $workingDir = $self->_workingDir->stringify;
  my $outDir = $self->outDir->stringify;

  if($self->archive) {
    my $supportFiles = join(",", grep { $_ !~ $self->outputFilesInfo->{annotation} } glob($self->_workingDir->child('*')->stringify));

    # First cp the support files, including statistics
    my $result = system("cp {$supportFiles} $outDir; sync");

    if($result) {
      $self->log('error', "Error copying support files: $!");
    }

    my $compressErr = $self->compressDirIntoTarball( $self->_workingDir, $self->outputFilesInfo->{archived} );

    if($compressErr) {
      return $compressErr;
    }
  }

  if( $self->outDir eq $self->_workingDir) {
    $self->log('debug', "Nothing to move, workingDir equals outDir");
    return 0;
  }

  $self->log('info', "Moving output file to EFS or S3");

  my $result = system("mv $workingDir/* $outDir; sync");

  return $result ? $! : 0;
}

# Replaces periods with _
# Database like Mongodb don't like periods
# Modifies the array
sub _normalizeSampleNames {
  my ($self, $inputHeader, $sampleIndicesAref) = @_;

  for my $idx (@$sampleIndicesAref) {
    $inputHeader->[$idx] =~ tr/./_/;
  }

  return $inputHeader;
}

# If we need another way of instantiating workingDir that is less error-prone
# (because of the extreme dependence on laziness)
around 'BUILD' => sub {
  my $orig = shift;
  my $self = shift;

  # Ensure that the output directory exists
  $self->outDir->mkpath();

  $self->$orig(@_);

  if($self->archive && !$self->temp_dir) {
    $self->log('fatal', "If you wish to 'archive', must specify 'temp_dir'");
  }
};

no Mouse::Role;
1;