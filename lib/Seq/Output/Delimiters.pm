use 5.10.0;
use strict;
use warnings;

package Seq::Output::Delimiters;
use Mouse 2;

has valueDelimiter => (is => 'ro', default => ';');

has positionDelimiter => (is => 'ro', default => '|');

has alleleDelimiter => (is => 'ro', default => '/');

has fieldSeparator => (is => 'ro', default => "\t");

has emptyFieldChar => (is => 'ro', default => "!");

__PACKAGE__->meta->make_immutable();
return 1;