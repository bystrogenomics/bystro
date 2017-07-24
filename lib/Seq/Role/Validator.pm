## Interface Class
use 5.10.0;

package Seq::Role::Validator;

use Mouse::Role;
use namespace::autoclean;

#also prrovides ->is_file function
use Types::Path::Tiny qw/File AbsFile AbsPath AbsDir/;

use DDP;

use Path::Tiny;
use Cwd 'abs_path';

use YAML::XS;
use Archive::Extract;
use Try::Tiny;
use File::Which;
use Carp qw(cluck confess);

use Seq::InputFile;

with 'Seq::Role::IO', 'Seq::Role::Message';

has assembly => (is => 'ro', required => 1);

has _inputFileBaseName => (
  isa => 'Str',
  is => 'ro',
  init_arg => undef,
  required => 0,
  lazy => 1,
  default => sub {
    my $self = shift;
    return $self->snpfile->basename(qr/\..*/);
  },
);

sub validateInputFile {
  my ( $self, $inputFileAbsPath ) = @_;

  if(!ref $inputFileAbsPath) {
    $inputFileAbsPath = path($inputFileAbsPath);
  }

  my $fh = $self->get_read_fh($inputFileAbsPath);
  my $firstLine = <$fh>;

  my $headerFieldsAref = $self->getCleanFields($firstLine);

  my $inputHandler = Seq::InputFile->new();

  #last argument to not die, we want to be able to convert
  my $snpHeaderErr = $inputHandler->checkInputFileHeader($headerFieldsAref, 1);

  if(!defined $headerFieldsAref || defined $snpHeaderErr) {
    return (0, 'vcf');
  }

  return (0, 'snp');
}

1;
