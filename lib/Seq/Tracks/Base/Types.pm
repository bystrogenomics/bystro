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
use Scalar::Util qw/looks_like_number/;

#What the types must be called in the config file
# TODO: build these track maps automatically
# by title casing the "type" field
# And therefore maybe don't use these at all.
state $refType = 'reference';
has refType => (is => 'ro', init_arg => undef, lazy => 1, default => sub{$refType});

state $scoreType = 'score';
has scoreType => (is => 'ro', init_arg => undef, lazy => 1, default => sub{$scoreType});

state $sparseType = 'sparse';
has sparseType => (is => 'ro', init_arg => undef, lazy => 1, default => sub{$sparseType});

state $regionType = 'region';
has regionType => (is => 'ro', init_arg => undef, lazy => 1, default => sub{$regionType});

state $geneType = 'gene';
has geneType => (is => 'ro', init_arg => undef, lazy => 1, default => sub{$geneType});

state $caddType = 'cadd';
has caddType => (is => 'ro', init_arg => undef, lazy => 1, default => sub{$caddType});

has trackTypes => (is => 'ro', init_arg => undef, lazy => 1, default => sub{
  return [$refType, $scoreType, $sparseType, $regionType, $geneType, $caddType]
});

enum TrackType => [$refType, $scoreType, $sparseType, $regionType, $geneType, $caddType];

#Convert types; Could move the conversion code elsewehre,
#but I wanted types definition close to implementation

enum DataType => ['float', 'int', 'number'];

#idiomatic way to re-use a stack, gain some efficiency
#expects ->convert('string or number', 'type')
sub convert {
  goto &{$_[2]}; #2nd argument, with $self == $_[0]
}

#For numeric types we need to check if we were given a weird string
#not certain if we should return, warn, or what
#in bioinformatics it seems very common to use "NA" or a "." or "-" to
#depict missing data

#@param {Str | Num} $_[1] : the data
#Note that if "-1.000000" is passed, it is NOT guaranteed to be returned as a float
#Strangely enough, it seems to be handled internally as an int potentially
#Or this could be dependent on msgpack-perl
sub float {
  if (!looks_like_number($_[1] ) ) {
    return $_[1];
  }

  return $_[1] + 0;
}

#@param {Str | Num} $_[1] : the data
sub int {
  if (!looks_like_number($_[1] ) ) {
    return $_[1];
  }

  #Truncates, doesn't round
  return CORE::int($_[1]);
}

# This is useful, because will convert a string like "1.000000" to an int
# And this will be interpreted in msgpack as an int, rather than a long string
# Similarly, all numbers *should* be storable within 9 bytes (float64),
# whereas if we sprintf, they will be stored as strings
sub number {
  if (!looks_like_number($_[1] ) ) {
    return $_[1];
  }

  #Saves us up to 8 bytes, because otherwise msgpack will store everything
  #as a 9 byte double
  if(CORE::int($_[1]) == $_[1]) {
    return CORE::int($_[1]);
  }

  return 0 + $_[1];
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