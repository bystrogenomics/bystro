use 5.10.0;
use strict;
use warnings;

package Seq::Tracks::Vcf;

our $VERSION = '0.001';

# ABSTRACT: The getter for any type == vcf track
# VERSION

use Mouse 2;
use namespace::autoclean;

extends 'Seq::Tracks::Get';

__PACKAGE__->meta->make_immutable;

sub BUILD {
  my $self = shift;
  $self->{_altIdx} = $self->getFieldDbName('alt');

  if(!defined $self->{_altIdx}) {
    die "Couldn't find 'alt' feature, required for Vcf tracks";
  }
}

sub get {
  # Avoid assignments, save overhead
  #my ($self, $href, $chr, $refBase, $allele, $alleleIdx, $positionIdx, $outAccum) = @_;
  #$href is the data that the user previously grabbed from the database
  # $_[0] == $self
  # $_[1] == $href : the array fetched for this position from the database
  # $_[2] == $chr
  # $_[3] == $refBase
  # $_[4] == $allele
  # $_[5] == $alleleIdx
  # $_[6] == $positionIdx
  # $_[7] == $outAccum

  my $data = $_[1]->[$_[0]->{_dbName}];
  my $alt = $data->[$_[0]->{_altIdx}];

  # If $alt is a reference (if not to array will die an ugly death, means horribly corrrupted db)
  # then find the matching alt, record the index, and look up all field values
  # at this index
  # All fields are required to have the same depth, during building
  # We attempt to only enter a for loop when necessary, because perl is slow
  # Most sites will not have overlapping records
  if (ref $alt) {
    my $dataIdx = 0;

    for my $alt (@$alt) {
      if($alt eq $_[4]) {
        for my $fieldIdx (@{$_[0]->{_fieldIdxRange}}) {
          #$outAccum->[$fieldIdx][$alleleIdx][$positionIdx] = $data->[$self->{_fieldDbNames}[$dataIdx]] }
          $_[7]->[$fieldIdx][$_[5]][$_[6]] = $data->[$_[0]->{_fieldDbNames}[$fieldIdx]][$dataIdx];
        }

        #return $outAccum;
        return $_[7];
      }

      $dataIdx++;
    }

    # If we got to this point, we found nothing.
    # Return nothing. That is a perfectly valid value
    # The output module will correctly write undefined values for all encessary
    # fields
    #return $outAccum;
    return $_[7];
  }

  # Alt is a scalar, which means there were no overlapping database values
  # at this pposiiton, and all fields represent a single value 
  for my $idx (@{$_[0]->{_fieldIdxRange}}) {
    #$outAccum->[$idx][$alleleIdx][$positionIdx] = $href->[$self->{_dbName}][$self->{_fieldDbNames}[$idx]] }
    $_[7]->[$idx][$_[5]][$_[6]] = $data->[$_[0]->{_fieldDbNames}[$idx]];
  }

  #return $outAccum
  return $_[7];
}

__PACKAGE__->meta->make_immutable;

1;
