use 5.10.0;
use strict;
use warnings;

package Seq::Role::ConfigFromFile;

our $VERSION = '0.001';

# ABSTRACT: A moose role for configuring a class from a YAML file
# VERSION

=head1 DESCRIPTION

  @role Seq::Role::ConfigFromFile
  #TODO: Check description

  @example with 'Seq::Role::ConfigFromFile'

Used in:
=for :list
* Seq::Annotate
* Seq::Assembly
* Seq::Fetch

Extended by: None

=cut
use Mouse::Role 2;

use Carp qw/ croak /;
use namespace::autoclean;
use Type::Params qw/ compile /;
use Types::Standard qw/ :types /;
use Scalar::Util qw/ reftype /;
use YAML::XS qw/ LoadFile /;
use DDP;

with 'Seq::Role::IO', 'MouseX::Getopt';

state $tracksKey = 'tracks';
#The only "Trick" added here is that we take everything that is outside the
#"tracks" key, and push that stuff in each tracks array item
#THe logic is that our YAML config file has 2 levels of content
#1) Global : key => value pairs that apply to every track
#2) Track-level : key => value pairs that only apply to that track
sub new_with_config {
  state $check = compile( Str, HashRef );
  my ( $class, $opts ) = $check->(@_);
  my %opts;

  my $config = $opts->{config};

  if ( !defined $config ) {
    croak "new_with_config() expects config";
  }

  my $hash = LoadFile($config);
  %opts = ( %$hash, %$opts );

  #Now push every single global option into each individual track
  #Since they are meant to operate as independent units
  my @nonTrackKeys = grep { $_ ne $tracksKey } keys %opts;

  if( ref $opts{$tracksKey} ne 'ARRAY') {
    croak "expect $tracksKey to contain an array of data";
  }

  for my $trackHref ( @{ $opts{$tracksKey} } ) {
    for my $key (@nonTrackKeys) {
      $trackHref->{$key} = $opts{$key};
    }
  }

  if ( $opts{debug} ) {
    say "Data for Role::ConfigFromFile::new_with_config()";
    p %opts;
  }

  $class->new( \%opts );
}

no Mouse::Role;

1;
