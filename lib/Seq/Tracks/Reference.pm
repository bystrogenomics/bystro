use 5.10.0;
use strict;
use warnings;

package Seq::Tracks::Reference;

our $VERSION = '0.001';

# ABSTRACT: The getter for the reference track
# VERSION

use Mouse 2;
use DDP;

use namespace::autoclean;

use Seq::Tracks::Reference::MapBases;

state $baseMapper = Seq::Tracks::Reference::MapBases->new();
state $baseMapInverse = $baseMapper->baseMapInverse;

extends 'Seq::Tracks::Get';

sub get {
  # $_[0] == $self; $_[1] = dbDataAref
  # $self->{_dbName} inherited from Seq::Tracks::Get
  # not declared here because putting in a builder here results in 
  # "Oops Destroying Active Enviroment in LMDB_File
  return $baseMapInverse->{ $_[1]->[ $_[0]->{_dbName} ] };
}

__PACKAGE__->meta->make_immutable;

1;
