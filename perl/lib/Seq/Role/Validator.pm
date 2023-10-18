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

has _inputFileBaseName => (
  isa      => 'Str',
  is       => 'ro',
  init_arg => undef,
  required => 0,
  lazy     => 1,
  default  => sub {
    my $self = shift;
    return $self->snpfile->basename(qr/\..*/);
  },
);

sub validateInputFile {
  my ( $self, $inputFilePath ) = @_;

  my @parts = split( "/", $inputFilePath );

  my $last = $parts[-1];

  # TODO: support more types
  for my $type ( ( "vcf", "snp" ) ) {
    my ( $format, $gz ) = $last =~ /\.($type)(\.\w+)?/;

    if ($format) {
      return ( 0, lc($format) );
    }
  }

  return ( "Couldn't identify format of $inputFilePath", "" );
}
1;
