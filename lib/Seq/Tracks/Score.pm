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

has scalingFactor => (is => 'ro', isa => 'Int', default => 100);

sub BUILD {
  my $self = shift;
  $self->{_s} = $self->scalingFactor;
}

sub get {
  #my ($self, $href, $chr, $refBase, $allele, $outAccum, $alleleNumber) = @_
  # $_[0] == $self
  # $_[1] == <ArrayRef> $href : the database data, with each top-level index corresponding to a track
  # $_[2] == <String> $chr  : the chromosome
  # $_[3] == <String> $refBase : ACTG
  # $_[4] == <String> $allele  : the allele (ACTG or -N / +ACTG)
  # $_[5] == <Int> $alleleIdx  : if this is a single-line multiallelic, the allele index
  # $_[6] == <Int> $positionIdx : the position in the indel, if any
  # $_[7] == <ArrayRef> $outAccum : a reference to the output, which we mutate

  $_[7][$_[6]] = defined $_[1]->[ $_[0]->{_dbName} ] ?  $_[1]->[ $_[0]->{_dbName} ] / $_[0]->{_s} : undef;

  return $_[7];
}


__PACKAGE__->meta->make_immutable;
1;
