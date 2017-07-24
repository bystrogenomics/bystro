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

#accepts $self, $dataHref, $chr (not used), $altAlleles
#@param <String|ArrayRef> $altAlleles : the alleles, like A,C,T,G or ['A','C','T','G'] 
sub get {
  #my ($self, $href, $chr, $refBase, $altAlleles, $outAccum, $alleleNumber) = @_
  # $_[0] == $self
  # $_[1] == $href
  # $_[2] == $chr
  # $_[3] == $refBase
  # $_[4] == $altAlleles
  # $_[5] == $alleleIdx
  # $_[6] == $positionIdx
  # $_[7] == $outAccum

  if (!defined $order->{$_[3]} ) {
    $_[0]->log('warn', "reference base $_[3] doesn't look valid, in Cadd.pm");
    
    $_[7][$_[5]][$_[6]] = undef;

    return $_[7];
  }

  # We may have stored an empty array at this position, in case 
  # the CADD scores read were not guaranteed to be sorted
  # Alternatively the CADD data for this position may be missing (not defined)
  if(!defined $_[1]->[$_[0]->{_dbName}] || !@{$_[1]->[$_[0]->{_dbName}]}) {
    $_[7][$_[5]][$_[6]] = undef;

    return $_[7];
  }
  
  #if (defined $order->{ $refBase }{ $altAlleles } ) {
  if ( defined $order->{$_[3]}{$_[4]} ) {
    $_[7][$_[5]][$_[6]] = $_[1]->[$_[0]->{_dbName}][ $order->{$_[3]}{$_[4]} ];

    return $_[7];
  }

  # For indels, which will be the least frequent, return it all
  if (length( $_[4] ) > 1) {
    $_[7][$_[5]][$_[6]] = $_[1]->[ $_[0]->{_dbName} ];

    return $_[7];
  }

  # Allele isn't an indel, but !defined $order->{ $refBase }{ $altAlleles }
  $_[7][$_[5]][$_[6]] = undef;

  return $_[7];
}

__PACKAGE__->meta->make_immutable;
1;