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
extends 'Seq::Tracks::Get';

state $order = Seq::Tracks::Cadd::Order->new();
$order = $order->order;

has scalingFactor => (is => 'ro', isa => 'Int', default => 10);

override 'BUILD' => sub {
  my $self = shift;

  # purely to save accessor time
  $self->{_s} = $self->scalingFactor;
  $self->{_d} = $self->dbName;
};

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

  if (!defined $order->{$_[3]} ) {
    $_[0]->log('warn', "reference base $_[3] doesn't look valid, in Cadd.pm");
    
    $_[7][$_[6]] = undef;

    return $_[7];
  }

  # We may have stored an empty array at this position, in case 
  # the CADD scores read were not guaranteed to be sorted
  # Alternatively the CADD data for this position may be missing (not defined)
  if(!defined $_[1]->[$_[0]->{_d}] || !@{$_[1]->[$_[0]->{_d}]}) {
    $_[7][$_[6]] = undef;

    return $_[7];
  }
  
  #if (defined $order->{ $refBase }{ $altAlleles } ) {
  if (defined $order->{$_[3]}{$_[4]} ) {
    $_[7][$_[6]] = $_[1]->[$_[0]->{_d}][ $order->{$_[3]}{$_[4]} ] / $_[0]->{_s};

    return $_[7];
  }

  # For indels, which will be the least frequent, return it all
  if (length( $_[4] ) > 1) {
    $_[7][$_[6]] = [ map { $_ / $_[0]->{_s} } @{$_[1]->[ $_[0]->{_d} ]} ];

    return $_[7];
  }

  # Allele isn't an indel, but !defined $order->{ $refBase }{ $altAlleles }
  $_[7][$_[6]] = undef;

  return $_[7];
}

__PACKAGE__->meta->make_immutable;
1;