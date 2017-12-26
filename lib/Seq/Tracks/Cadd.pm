use 5.10.0;
use strict;
use warnings;

# TODO: refactor to allow mutliple alleles, and multiple posiitions
package Seq::Tracks::Cadd;

#A track whose features are only reported if they match the minor allele 
#present in the sample
#Called cadd because at the time of writing it's the 
use Mouse 2;
use namespace::autoclean;
use Seq::Tracks::Cadd::Order;

# This esesntially is a score track, just needs to lookup the value in the array
extends 'Seq::Tracks::Score';

state $order = Seq::Tracks::Cadd::Order->new();
$order = $order->order;

# CADD scores use (by default), a lower scaling factor, because
# We expet cadd scores < 10 to be wholle irrelevant
# and CADD scores > 15 more or less the same
# From the 2013 paper, maximum discrimination occurs at 15
# So we more or less don't care about the decimal; keeping it solely to give some
# ability to break ties
has '+scalingFactor' => (default => 10);

#Provided by Seq::Tracks::Get
#$self->{_dbName} = $self->dbName;
sub get {
  #my ($self, $href, $chr, $refBase, $allele, $outAccum, $alleleNumber) = @_
  # $_[0] == $self
  # $_[1] == <ArrayRef> $href : the database data, with each top-level index corresponding to a track
  # $_[2] == <String> $chr  : the chromosome
  # $_[3] == <String> $refBase : ACTG
  # $_[4] == <String> $allele  : the allele (ACTG or -N / +ACTG)
  # $_[5] == <Int> $positionIdx : the position in the indel, if any
  # $_[6] == <ArrayRef> $outAccum : a reference to the output, which we mutate

  # We may have stored an empty array at this position, in case
  # the CADD scores read were not guaranteed to be sorted
  # Alternatively the CADD data for this position may be missing (not defined)
  # It's slightly faster to check for truthiness, rather than definition
  # Since we always either store undef or an array, truthiness is sufficient
  if(!$_[1]->[$_[0]->{_dbName}]) {
    $_[6][$_[5]] = undef;

    return $_[6];
  }

  if (!$order->{$_[3]} ) {
    $_[0]->log('warn', "reference base $_[3] doesn't look valid, in Cadd.pm");
    
    $_[6][$_[5]] = undef;

    return $_[6];
  }

  #if (defined $order->{ $refBase }{ $altAlleles } ) {
  if ($order->{$_[3]}{$_[4]} ) {
    $_[6][$_[5]] = $_[1]->[$_[0]->{_dbName}][ $order->{$_[3]}{$_[4]} ] / $_[0]->{_s};

    return $_[6];
  }

  # For indels, which will be the least frequent, return it all
  if (length( $_[4] ) > 1) {
    $_[6][$_[5]] = [ map { $_ / $_[0]->{_s} } @{$_[1]->[ $_[0]->{_dbName} ]} ];

    return $_[6];
  }

  # Allele isn't an indel, but !defined $order->{ $refBase }{ $altAlleles }
  $_[6][$_[5]] = undef;

  return $_[6];
}

__PACKAGE__->meta->make_immutable;
1;