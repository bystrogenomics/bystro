use 5.10.0;
use strict;
use warnings;

package Seq::Tracks::Gene::Nearest::Definition;
use Mouse::Role 2;
#Defines a few keys common to the build and get functions of Tracks::Gene

# TODO: at some point allow exonStarts, exonEnds, and potentially
# any field names, with some configuration.
enum ValidNearestAnchors => ['txStart','txEnd','cdsStart','cdsEnd'];

# Coordinate we start looking from
has from => (is => 'ro', isa => 'ValidNearestAnchors', default => 'txStart');

# Coordinate we look to
has to => (is => 'ro', isa => 'ValidNearestAnchors', default => 'txEnd');

no Mouse::Role;
1;