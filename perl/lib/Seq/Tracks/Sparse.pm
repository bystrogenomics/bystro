use 5.10.0;
use strict;
use warnings;

package Seq::Tracks::Sparse;

our $VERSION = '0.001';

# ABSTRACT: The getter for any sparse track
# VERSION

use Mouse 2;
use namespace::autoclean;

extends 'Seq::Tracks::Get';

__PACKAGE__->meta->make_immutable;

1;
