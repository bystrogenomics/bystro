use 5.10.0;
use strict;
use warnings;

package Seq::Tracks::Reference::MapBases;
use Mouse 2;
use namespace::autoclean;

# Abstract: Maps bases to integers, which saves space in the db
state $baseMap = { N => 0, A => 1, C => 2, G => 3, T => 4 };
has baseMap => ( is => 'ro', init_arg => undef, lazy => 1, default => sub{ $baseMap });

has baseMapInverse => ( is => 'ro', init_arg => undef, lazy => 1, default => sub {
  return { map { $baseMap->{$_} => $_ } keys %$baseMap };
});

__PACKAGE__->meta->make_immutable;
1;
