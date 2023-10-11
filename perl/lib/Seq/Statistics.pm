use 5.10.0;
use strict;
use warnings;

package Seq::Statistics;

use Mouse 2;
use namespace::autoclean;

use Seq::Output::Delimiters;
use Types::Path::Tiny qw/AbsPath AbsFile AbsDir/;
use File::Which qw/which/;

with 'Seq::Role::Message';

############################ Exports #########################################
has jsonFilePath => (is => 'ro', init_arg => undef, writer => '_setJsonFilePath');

has tabFilePath => (is => 'ro', init_arg => undef, writer => '_setTabFilePath');

has qcFilePath => (is => 'ro', init_arg => undef, writer => '_setQcFilePath');


################################# Required ################################
# The path to the output file without extensions
has outputBasePath => (is => 'ro', isa => 'Str', required => 1);

# Comes from YAML or command line: what bystro features are called
has dbSNPnameField => (is => 'ro', isa => 'Str', default => '');

# Comes from YAML or command line: what bystro features are called
has siteTypeField => (is => 'ro', isa => 'Str', required => 1);

# Comes from YAML or command line: The tracks configuration
has exonicAlleleFunctionField => (is => 'ro', isa => 'Str', required => 1);

# Optional. Can be specified in YAML, command line, or passed.
# If not passed, 
has refTrackField => (is => 'ro', isa => 'Str', required => 1);

has altField => (is => 'ro', isa => 'Str', required => 1);

has homozygotesField => (is => 'ro', isa => 'Str', required => 1);

has heterozygotesField => (is => 'ro', isa => 'Str', required => 1);

############################### Optional ##################################

# The statistics package config options
# This is by default the go program we use to calculate statistics
has programPath => (is => 'ro', isa => 'Str', default => 'bystro-stats');

has outputExtensions => (is => 'ro', isa => 'HashRef', default => sub {
  return {
    json => '.statistics.json',
    tab => '.statistics.tab',
    qc => '.statistics.qc.tab',
  }
});

# TODO: Store error, return that from BUILD, instead of err
sub BUILDARGS {
  my ($self, $data) = @_;

  if(defined $data->{outputExtensions}) {
    if(!($data->{outputExtensions}{json} && $data->{outputExtensions}{qc} && $data->{outputExtensions}{tab})) {
      $self->log('fatal', "outputExtensions property requires json, qc, tab values");
      return;
    }
  }

  return $data;
}

sub BUILD {
  my $self = shift;

  $self->{_delimiters} = Seq::Output::Delimiters->new();
  
  #TODO: Should we store the which path at $self->statistics_program
  if (!which($self->programPath)) {
    $self->log('fatal', "Couldn't find statistics program at " . $self->programPath);
    return;
  }

  $self->_setJsonFilePath($self->outputBasePath . $self->outputExtensions->{json});
  $self->_setTabFilePath($self->outputBasePath . $self->outputExtensions->{tab});
  $self->_setQcFilePath($self->outputBasePath . $self->outputExtensions->{qc});
}

sub getStatsArguments {
  my $self = shift;

  # Accumulate the delimiters: Note that $alleleDelimiter isn't necessary
  # because the bystro_statistics script never operates on multiallelic sites
  my $valueDelimiter = $self->{_delimiters}->valueDelimiter;

  my $fieldSeparator = $self->{_delimiters}->fieldSeparator;
  my $emptyFieldString = $self->{_delimiters}->emptyFieldChar;

  my $refColumnName = $self->refTrackField;
  my $alleleColumnName = $self->altField;

  my $homozygotesColumnName = $self->homozygotesField;
  my $heterozygotesColumnName = $self->heterozygotesField;

  my $jsonOutPath = $self->jsonFilePath;
  my $tabOutPath = $self->tabFilePath;
  my $qcOutPath = $self->qcFilePath;

  my $siteTypeColumnName = $self->siteTypeField;
  my $snpNameColumnName = $self->dbSNPnameField;
  my $exonicAlleleFuncColumnName = $self->exonicAlleleFunctionField;

  my $statsProg = which($self->programPath);

  if (!$statsProg && ($snpNameColumnName && $exonicAlleleFuncColumnName && $emptyFieldString && $valueDelimiter
  && $refColumnName && $alleleColumnName && $siteTypeColumnName && $homozygotesColumnName
  && $heterozygotesColumnName && $jsonOutPath && $tabOutPath && $qcOutPath)) {
    return ("Need, refColumnName, alleleColumnName, siteTypeColumnName, homozygotesColumnName,"
      . "heterozygotesColumnName, jsonOutPath, tabOutPath, qcOutPath, "
      . "primaryDelimiter, fieldSeparator, and "
      . "numberHeaderLines must equal 1 for statistics", undef, undef);
  }

  my $dbSNPpart = "";

  if($snpNameColumnName) {
    $dbSNPpart = "-dbSnpNameColumn $snpNameColumnName ";
  }

  return (undef, "$statsProg -outJsonPath $jsonOutPath -outTabPath $tabOutPath "
    . "-outQcTabPath $qcOutPath -refColumn $refColumnName "
    . "-altColumn $alleleColumnName -homozygotesColumn $homozygotesColumnName "
    . "-heterozygotesColumn $heterozygotesColumnName -siteTypeColumn $siteTypeColumnName "
    . $dbSNPpart
    . "-emptyField '$emptyFieldString' "
    . "-exonicAlleleFunctionColumn $exonicAlleleFuncColumnName "
    . "-primaryDelimiter '$valueDelimiter' -fieldSeparator '$fieldSeparator' ");
}

__PACKAGE__->meta->make_immutable;

1;