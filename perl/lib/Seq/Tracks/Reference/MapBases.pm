use 5.10.0;
use strict;
use warnings;

package Seq::Tracks::Reference::MapBases;
use Mouse 2;
use namespace::autoclean;

# Abstract: Maps bases to integers, which saves space in the db
# Encodes lowercase or uppercase letts into 0 - 4 byte, returns only uppercase letters
state $baseMap = {
  N => 0,
  A => 1,
  C => 2,
  G => 3,
  T => 4,
  n => 0,
  a => 1,
  c => 2,
  g => 3,
  t => 4
};
has baseMap => (
  is       => 'ro',
  isa      => 'HashRef',
  init_arg => undef,
  lazy     => 1,
  default  => sub { $baseMap }
);

has baseMapInverse => (
  is       => 'ro',
  isa      => 'ArrayRef',
  init_arg => undef,
  lazy     => 1,
  default  => sub {
    return [ 'N', 'A', 'C', 'G', 'T' ];
  }
);

__PACKAGE__->meta->make_immutable;
1;
