use 5.10.0;
use strict;
use warnings;

package Seq::Tracks::Base::Types;

our $VERSION = '0.001';

# ABSTRACT: Defines general track information: valid track "types",
# track casting (data) types
# VERSION

use Mouse 2;
use Mouse::Util::TypeConstraints;
use namespace::autoclean;
use Scalar::Util  qw/looks_like_number/;
use Math::SigFigs qw(:all);

#What the types must be called in the config file
# TODO: build these track maps automatically
# by title casing the "type" field
# And therefore maybe don't use these at all.
state $refType = 'reference';
has refType =>
  ( is => 'ro', init_arg => undef, lazy => 1, default => sub { $refType } );

state $scoreType = 'score';
has scoreType =>
  ( is => 'ro', init_arg => undef, lazy => 1, default => sub { $scoreType } );

state $sparseType = 'sparse';
has sparseType =>
  ( is => 'ro', init_arg => undef, lazy => 1, default => sub { $sparseType } );

state $regionType = 'region';
has regionType =>
  ( is => 'ro', init_arg => undef, lazy => 1, default => sub { $regionType } );

state $geneType = 'gene';
has geneType =>
  ( is => 'ro', init_arg => undef, lazy => 1, default => sub { $geneType } );

state $caddType = 'cadd';
has caddType =>
  ( is => 'ro', init_arg => undef, lazy => 1, default => sub { $caddType } );

state $vcfType = 'vcf';
has vcfType =>
  ( is => 'ro', init_arg => undef, lazy => 1, default => sub { $vcfType } );

state $nearestType = 'nearest';
has nearestType =>
  ( is => 'ro', init_arg => undef, lazy => 1, default => sub { $nearestType } );

has trackTypes => (
  is       => 'ro',
  init_arg => undef,
  lazy     => 1,
  default  => sub {
    return [ $refType, $scoreType, $sparseType, $regionType, $geneType, $caddType,
      $vcfType ];
  }
);

enum TrackType => [
  $refType,  $scoreType, $sparseType, $regionType,
  $geneType, $caddType,  $vcfType,    $nearestType
];

#Convert types; Could move the conversion code elsewehre,
#but I wanted types definition close to implementation

subtype DataType => as 'Str' =>
  where { $_ =~ /number|number\(\d+\)/ }; #['float', 'int', 'number', 'number(2)'];

# float / number / int can give a precision in the form number(2)
state $precision = {};
state $typeFunc  = {};
#idiomatic way to re-use a stack, gain some efficiency
#expects ->convert('string or number', 'type')
sub convert {
  #my ($self, $value, $type)
  #    $_[0], $_[1],  $_[2]
  if ( !$typeFunc->{ $_[2] } ) {
    my $idx = index( $_[2], '(' );

    # We're given number(N) where N is sig figs
    if ( $idx > -1 ) {
      my $type = substr( $_[2], 0, $idx );

      $typeFunc->{ $_[2] } = \&{$type};
      # if number(2) take "2" + 0 == 2
      $precision->{ $_[2] } = substr( $_[2], $idx + 1, index( $_[2], ')' ) - $idx - 1 ) +0;
    }
    else {
      # We're given just "number", no precision, so use the type itself ($_[2])
      $typeFunc->{ $_[2] }  = \&{ $_[2] };
      $precision->{ $_[2] } = -1;
    }
  }

  return $typeFunc->{ $_[2] }->( $_[1], $precision->{ $_[2] } )
    ; #2nd argument, with $self == $_[0]
}

# Truncate a number
sub int {
  #my ($value, $precision) = @_;
  #    $_[0], $_[1],
  if ( !looks_like_number( $_[0] ) ) {
    return $_[0];
  }

  return CORE::int( $_[0] );
}

# This is useful, because will convert a string like "1.000000" to an int
# And this will be interpreted in msgpack as an int, rather than a long string
# Similarly, all numbers *should* be storable within 9 bytes (float64),
# whereas if we sprintf, they will be stored as strings
# Will always take the smallest possible value, so will only be stored as float
# if needed
sub number {
  #my ($value, $precision) = @_;
  #    $_[0], $_[1],
  if ( !looks_like_number( $_[0] ) ) {
    return $_[0];
  }

  #Saves us up to 8 bytes, because otherwise msgpack will store everything
  #as a 9 byte double
  if ( CORE::int( $_[0] ) == $_[0] ) {
    return CORE::int( $_[0] );
  }

  # Add 0 to prevent from being treated as string by serializers
  if ( $_[1] > 0 ) {
    return 0+ FormatSigFigs( $_[0], $_[1] );
  }

  # No precision given, just ducktype into a number
  return 0+ $_[0];
}

#moved away from this; the base build class shouldn't need to know
#what types are allowed, that info is kep in the various track modules
#this is a simple-minded way to enforce a bed-only format
#this should not be used for things with single-field headers
#like wig or multi-fasta (or fasta)
# enum BedFieldType => ['chrom', 'chromStart', 'chromEnd'];

no Mouse::Util::TypeConstraints;
__PACKAGE__->meta->make_immutable;

1;
