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

extends 'Seq::Tracks::Base';
with 'Seq::Tracks::Region::RegionTrackPath';

use Seq::Headers;
use Seq::DBManager;

# Coordinate we start looking from
# Typically used for gene tracks, would be 'txStart'
has from => ( is => 'ro', isa => 'Str', required => 1 );

# Coordinate we look to
# Typically used for gene tracks, would 'txEnd' or nothing
has to => ( is => 'ro', isa => 'Maybe[Str]' );

has dist => ( is => 'ro', isa => 'Bool', default => 1 );

#### Add our other "features", everything we find for this site ####
sub BUILD {
  my $self = shift;

  # Avoid accessor penalties in Mouse/Moose;

  # We only have 1 end
  $self->{_eq} = !$self->to || $self->from eq $self->to;
  # We expect to ALWAYS have a from field
  $self->{_fromD} = $self->getFieldDbName( $self->from );
  # But "to" is optional
  $self->{_toD} = !$self->{_eq} ? $self->getFieldDbName( $self->to ) : undef;

  $self->{_db}     = Seq::DBManager->new();
  $self->{_dbName} = $self->dbName;

  # We may or may not want to calculate distance
  $self->{_dist}         = !!$self->dist;
  $self->{_fieldDbNames} = [ map { $self->getFieldDbName($_) } @{ $self->features } ];
}

sub setHeaders {
  my $self = shift;

  my $headers  = Seq::Headers->new();
  my @features = @{ $self->features };

  if ( $self->dist ) {
    push @features, 'dist';
  }

  $headers->addFeaturesToHeader( \@features, $self->name );

  # If we have dist, it comes as our last feature
  # We don't have a field db name for dist...this is a calculated feature
  # so we just want to get into the header, and into the output
  $self->{_fIdx} = [ 0 .. $#{ $self->features } ];
}

sub get {
  #my ($self, $href, $chr, $refBase, $allele, $positionIdx, $outAccum, $position) = @_
  # $_[0] == $self
  # $_[1] == <ArrayRef> $href : the database data, with each top-level index corresponding to a track
  # $_[2] == <String> $chr  : the chromosome
  # $_[3] == <String> $refBase : ACTG
  # $_[4] == <String> $allele  : the allele (ACTG or -N / +ACTG)
  # $_[5] == <Int> $positionIdx : the position in the indel, if any
  # $_[6] == <ArrayRef> $outAccum : a reference to the output, which we mutate
  # $_[7] == <Int> $zeroPos : the 0-based genomic position
  ################# Cache track's region data ##############
  #$self->{_regionData}{$chr} //= $self->{_db}->dbReadAll( $self->regionTrackPath($_[2]) );

  # If the position idx isn't 0 we're in an indel
  # We should make a decision whether to tile across the genom
  # My opinion atm is its substantially easier to just consider the indel
  # from the starting position w.r.t nearest data
  # However, this also removes useful information when an indel spans
  # multiple regions (in our use case mostly genes)
  # if($_[5] != 0) {
  #   return $_[6];
  # }

  # WARNING: If $_[1]->[$_[0]->{_dbName} isn't defined, will be treated as the 0 index!!!
  # therefore return here if that is the case
  if ( !defined $_[1]->[ $_[0]->{_dbName} ] ) {
    for my $i ( @{ $_[0]->{_fIdx} } ) {
      $_[6]->[$i][ $_[5] ] = undef;
    }

    return $_[6];
  }

  $_[0]->{_regionData}{ $_[2] } //=
    $_[0]->{_db}->dbReadAll( $_[0]->regionTrackPath( $_[2] ) );

  my $geneDb = $_[0]->{_regionData}{ $_[2] }[ $_[1]->[ $_[0]->{_dbName} ] ];

  # exit;
  # We have features, so let's find those and return them
  # Since all features are stored in some shortened form in the db, we also
  # will first need to get their dbNames ($self->getFieldDbName)
  # and these dbNames will be found as a value of $href->{$self->dbName}
  # #http://ideone.com/WD3Ele

  # All features from overlapping are already combined into arrays, unlike
  # what gene tracks used to do
  # Here we accumulate all features, except for the dist (not included in _fieldDbNames)
  my $i = 0;
  for my $fieldDbName ( @{ $_[0]->{_fieldDbNames} } ) {
    #$outAccum->[$i][$positionIdx] = $href->[$self->{_dbName}][$self->{_fieldDbNames}[$i]] }
    $_[6]->[$i][ $_[5] ] = $geneDb->[$fieldDbName];
    $i++;
  }

  # Calculate distance if requested
  # We always expect from and to fields to be scalars
  # Notice that dist is our last feature, because $i incremented +1 here
  if ( $_[0]->{_dist} ) {
    if ( $_[0]->{_eq} || $_[7] < $geneDb->[ $_[0]->{_fromD} ] ) {
      # We're before the starting position of the nearest region
      # Or we're only checking one boundary (the from boundary)
      $_[6]->[$i][ $_[5] ] = $geneDb->[ $_[0]->{_fromD} ] - $_[7];
    }
    elsif ( $_[7] <= $geneDb->[ $_[0]->{_toD} ] ) {
      # We already know $zeroPos >= $geneDb->[$_[0]->{_fromD}]
      # so if we're here, we are within the range of the requested region at this position
      # ie == 0 distance to the region
      $_[6]->[$i][ $_[5] ] = 0;
    }
    else {
      # occurs after the 'to' position
      $_[6]->[$i][ $_[5] ] = $geneDb->[ $_[0]->{_toD} ] - $_[7];
    }
  }

  return $_[6];
}

__PACKAGE__->meta->make_immutable;

1;
