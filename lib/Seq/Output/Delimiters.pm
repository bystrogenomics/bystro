use 5.10.0;
use strict;
use warnings;

package Seq::Output::Delimiters;
use Mouse 2;
with 'Seq::Role::Message';

has valueDelimiter => (is => 'ro', isa => 'Str', default => ';');

has positionDelimiter => (is => 'ro', isa => 'Str', default => "\|");

has alleleDelimiter => (is => 'ro', isa => 'Str',  default => "\/");

has fieldSeparator => (is => 'ro', isa => 'Str', default => "\t");

has emptyFieldChar => (is => 'ro',  isa => 'Str', default => "!");

__PACKAGE__->meta->make_immutable();
return 1;