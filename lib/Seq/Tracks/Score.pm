use 5.10.0;
use strict;
use warnings;

package Seq::Tracks::Score;

our $VERSION = '0.001';

# ABSTRACT: The getter for any score track
# VERSION

use Mouse 2;
use namespace::autoclean;

extends 'Seq::Tracks::Get';

__PACKAGE__->meta->make_immutable;

1;
