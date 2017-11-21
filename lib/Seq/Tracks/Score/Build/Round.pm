use 5.10.0;
use strict;
use warnings;

package Seq::Tracks::Score::Build::Round;
use Mouse 2;
use Math::SigFigs qw/:all/;

#TODO: Allow configuratino through YAML
#
sub round {
  #my ($self, $value) = @_;
  # $value == $_[1]

  #If we have an exact figure, return an int, stored by msgpack as an int
  if( int($_[1]) == $_[1] ) {
    return int($_[1]);
  }

  #We have updated Data::MessagePack to support, enforce single-precision floats
  #So 5 bytes at most when prefer_float32() enabled
  return FormatSigFigs($_[1], 2) + 0;
}

__PACKAGE__->meta->make_immutable;
1;
