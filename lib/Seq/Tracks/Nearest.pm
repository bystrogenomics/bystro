use 5.10.0;
use strict;
use warnings;
package Seq::Tracks::Nearest;

our $VERSION = '0.001';

=head1 DESCRIPTION

  @class B<Seq::Gene>
  
  A Region track that also calculates distance if the user wishes
  IE it pulls any requested features by a region reference (an in 0 to N-1)
  And then if "dist" is specified, calculates that based on the "from" and "to" features

=cut

## TODO: remove the if(nearestGeneNumber) check. Right now needed because
## we have no chrM refSeq stuff
use Mouse 2;

use namespace::autoclean;
use DDP;

extends 'Seq::Tracks::Get';
with 'Seq::Tracks::Region::RegionTrackPath';

use Seq::DBManager;

# Coordinate we start looking from
# Typically used for gene tracks, would be 'txStart'
has from => (is => 'ro', isa => 'Str', required => 1);

# Coordinate we look to
# Typically used for gene tracks, would 'txEnd' or nothing
has to => (is => 'ro', isa => 'Maybe[Str]');

has dist => (is => 'ro', isa => 'Bool', default => 1);

#### Add our other "features", everything we find for this site ####
sub BUILD {
  my $self = shift;

  # Avoid accessor penalties in Mouse/Moose;
  $self->{_eq} = !$self->to || $self->from eq $self->to;
  $self->{_fromD} = $self->getFieldDbName($self->from);
  $self->{_toD} = !$self->{_eq} ? $self->getFieldDbName($self->to) : undef;
  $self->{_db} = Seq::DBManager->new();

  if($self->dist) {
    $self->{_dist} = 1;
    $self->headers->addFeaturesToHeader('dist', $self->name);
    push @{$self->{_i}}, $#{$self->features} + 1;
  }

  $self->{_fieldDbNames} = [ map { $self->getFieldDbName($_) } @{$self->features} ];

  $self->{_i} = [0 .. $#{$self->features}];
};

sub get {
  #my ($self, $href, $chr, $refBase, $allele, $positionIdx, $outAccum, $position) = @_
  # $_[0] == $self
  # $_[1] == <ArrayRef> $href : the database data, with each top-level index corresponding to a track
  # $_[2] == <String> $chr  : the chromosome
  # $_[3] == <String> $refBase : ACTG
  # $_[4] == <String> $allele  : the allele (ACTG or -N / +ACTG)
  # $_[5] == <Int> $alleleIdx  : if this is a single-line multiallelic, the allele index
  # $_[6] == <Int> $positionIdx : the position in the indel, if any
  # $_[7] == <ArrayRef> $outAccum : a reference to the output, which we mutate
  # $_[8] == <Int> $position : the genomic position
  ################# Cache track's region data ##############
  #$self->{_regionData}{$chr} //= $self->{_db}->dbReadAll( $self->regionTrackPath($_[2]) );

  # If the position idx isn't 0 we're in an indel
  # We should make a decision whether to tile across the genom
  # My opinion atm is its substantially easier to just consider the indel
  # from the starting position w.r.t nearest data
  # However, this also removes useful information when an indel spans
  # multiple regions (in our use case mostly genes)
  # if($_[6] != 0) {
  #   return $_[7];
  # }

  # WARNING: If $_[1]->[$_[0]->{_dbName} isn't defined, will be treated as the 0 index!!!
  # therefore return here if that is the case
  if(!defined $_[1]->[$_[0]->{_dbName}]) {
    for my $i (@{$_[0]->{_i}}) {
      $_[7]->[$i][$_[6]] = undef;
    }

    return $_[7];
  }

  $_[0]->{_regionData}{$_[2]} //= $_[0]->{_db}->dbReadAll( $_[0]->regionTrackPath($_[2]) );

  # WARNING: If $_[1]->[$_[0]->{_dbName} isn't defined, will be treated as the 0 index!!!
  #            $self->{_regionData}{$_[2]}[$href->[$self->{_dbName}}]];
  my $geneDb = $_[0]->{_regionData}{$_[2]}[$_[1]->[$_[0]->{_dbName}]];

  # exit;
  # We have features, so let's find those and return them
  # Since all features are stored in some shortened form in the db, we also
  # will first need to get their dbNames ($self->getFieldDbName)
  # and these dbNames will be found as a value of $href->{$self->dbName}
  # #http://ideone.com/WD3Ele

  # All features from overlapping are already combined into arrays, unlike
  # what gene tracks used to do
  my $idx = 0;
  for my $fieldDbName (@{$_[0]->{_fieldDbNames}}) {
    #$outAccum->[$idx][$alleleIdx][$positionIdx] = $href->[$self->{_dbName}][$self->{_fieldDbNames}[$idx]] }
    $_[7]->[$idx][$_[6]] = $geneDb->[$fieldDbName];
    $idx++;
  }

  # Calculate distance if requested
  # We always expect from and to fields to be scalars
  if($_[0]->{_dist}) {
    if($_[0]->{_eq} || $_[8] < $geneDb->[$_[0]->{_fromD}]) {
      # We're before the starting position of the nearest region
      # Or we're only checking one boundary (the from boundary)
      $_[7]->[$idx][$_[6]] = $geneDb->[$_[0]->{_fromD}] - $_[8];
    } elsif($_[8] <= $geneDb->[$_[0]->{_toD}]) {
      # We already know $_[8] >= $geneDb->[$_[0]->{_fromD}]
      # so if we're here, we are within the range of the requested region at this position
      # ie == 0 distance to the region
      $_[7]->[$idx][$_[6]] = 0
    } else {
      # occurs after the 'to' position
      $_[7]->[$idx][$_[6]] = $geneDb->[$_[0]->{_toD}] - $_[8];
    }
  }

  return $_[7];
};

__PACKAGE__->meta->make_immutable;

1;