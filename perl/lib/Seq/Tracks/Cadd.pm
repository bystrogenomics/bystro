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
extends 'Seq::Tracks::Get';

has scalingFactor => ( is => 'ro', isa => 'Int', default => 10 );

sub BUILD {
  my $self = shift;

  # purely to save accessor time
  $self->{_s} = $self->scalingFactor;

  #Provided by Seq::Tracks::Get
  #$self->{_dbName} = $self->dbName;
}

state $order = Seq::Tracks::Cadd::Order->new();
$order = $order->order;

sub get {
  #my ($self, $href, $chr, $refBase, $allele, $outAccum, $alleleNumber) = @_
  # $_[0] == $self
  # $_[1] == <ArrayRef> $href : the database data, with each top-level index corresponding to a track
  # $_[2] == <String> $chr  : the chromosome
  # $_[3] == <String> $refBase : ACTG
  # $_[4] == <String> $allele  : the allele (ACTG or -<length of deletion> / +ACTG)
  # $_[5] == <Int> $positionIdx : the position in the indel, if any
  # $_[6] == <ArrayRef> $outAccum : a reference to the output, which we mutate

  # We may have stored an empty array at this position, in case
  # the CADD scores read were not guaranteed to be sorted
  # Alternatively the CADD data for this position may be missing (not defined)
  # It's slightly faster to check for truthiness, rather than definition
  # Since we always either store undef or an array, truthiness is sufficient

  # Similar to VCF tracks, we only return exact matches
  # So we do not tile across the entire deleted / inserted region
  # We return only the first result
  # In the case of a SNP there is only 1 result
  # In the case of an indel, we return undef only once
  if ( $_[5] > 0 ) {
    return $_[6];
  }

  if ( !defined $_[1]->[ $_[0]->{_dbName} ] ) {
    $_[6][ $_[5] ] = undef;

    return $_[6];
  }

  # For indels, we return undef
  # Bystro always left-normalizes alleles, so the ref is always length 1
  # and alleles are always longer than 1 for indels
  # In the case of deletions, it is -<length of deletion>
  # In the case of insertions, it is +<inserted sequence>
  if ( length( $_[4] ) > 1 ) {
    $_[6][ $_[5] ] = undef;

    return $_[6];
  }

  if ( !defined $order->{ $_[3] } ) {
    $_[0]->log( 'warn', "reference base $_[3] doesn't look valid, in Cadd.pm" );

    $_[6][ $_[5] ] = undef;

    return $_[6];
  }

  #if (defined $order->{ $refBase }{ $altAlleles } ) {
  if ( defined $order->{ $_[3] }{ $_[4] } ) {
    $_[6][ $_[5] ] =
      $_[1]->[ $_[0]->{_dbName} ][ $order->{ $_[3] }{ $_[4] } ] / $_[0]->{_s};

    return $_[6];
  }

  # We match on exact allele
  $_[6][ $_[5] ] = undef;

  return $_[6];
}

__PACKAGE__->meta->make_immutable;
1;
