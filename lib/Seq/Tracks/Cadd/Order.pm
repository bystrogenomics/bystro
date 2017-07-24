use 5.10.0;
use strict;
use warnings;

package Seq::Tracks::Cadd::Order;
use Mouse 2;

state $order = {
  A => {
    C => 0,
    G => 1,
    T => 2,
  },
  C => {
    A => 0,
    G => 1,
    T => 2,
  },
  G => {
    A => 0,
    C => 1,
    T => 2,
  },
  T => {
    A => 0,
    C => 1,
    G => 2,
  },
  N => {
    A => 0,
    C => 1,
    G => 2,
    T => 3,
  }
};

has order => (is => 'ro', init_arg => undef, default => sub{$order});

__PACKAGE__->meta->make_immutable;
1;
