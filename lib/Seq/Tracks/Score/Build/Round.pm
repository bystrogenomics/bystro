use 5.10.0;
use strict;
use warnings;

package Seq::Tracks::Score::Build::Round;
use Mouse 2;
use POSIX qw/abs/;

#TODO: Combined this with Tracks::Build::MapTypes other conversion functions
sub round {
  #my ($self, $value) = @_;
  # $value == $_[1]

  #If we have an exact figure, return an int, stored by msgpack as an int
  if( int($_[1]) == $_[1] ) {
    return int($_[1]);
  }

  #If not, store 2 significant digits, and store as string, because
  #Data::MessagePack has broken float32 support (stores as float64)
  #This causes the return value to be a string, which is stored by msgpack
  #as a string
  if(abs($_[1]) > 10) {
    return sprintf "%0.1f", $_[1];
  }

  return sprintf "%0.2f", $_[1];
}

__PACKAGE__->meta->make_immutable;
1;


