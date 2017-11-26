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
has to => (is => 'ro', isa => 'Str');

has dist => (is => 'ro', isa => 'Bool', default => 1);

#### Add our other "features", everything we find for this site ####
sub BUILD {
  my $self = shift;
  # Not including the txNumberKey;  this is separate from the annotations, which is 
  # what these keys represent

  #  Prepend some custom features
  #  Providing 1 as the last argument means "prepend" instead of append
  #  So these features will come before any other refSeq.* features
  $self->headers->addFeaturesToHeader('dist', $self->name, 1);

  # Avoid accessor penalties in Mouse/Moose;
  $self->{_eq} = !$self->to || $self->from eq $self->to;
  $self->{_from} = $self->from;
  $self->{_to} = $self->to;
  $self->{_db} = Seq::DBManager->new();
  $self->{_dist} = $self->dist;
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
  $_[0]->{_regionData}{$_[2]} //= $_[0]->{_db}->dbReadAll( $_[0]->regionTrackPath($_[2]) );

  #            $self->{_regionData}{$_[2]}{$href->$self->{_dbName}}]};
  my $geneDb = $_[0]->{_regionData}{$_[2]}{$_[1]->[$_[0]->{_dbName}]};

  # We have features, so let's find those and return them
  # Since all features are stored in some shortened form in the db, we also
  # will first need to get their dbNames ($self->getFieldDbName)
  # and these dbNames will be found as a value of $href->{$self->dbName}
  # #http://ideone.com/WD3Ele
  # return [ map { $_[1]->[$_[0]->{_dbName}][$_] } @{$_[0]->{_fieldDbNames}} ];

  my $idx = 0;
  for my $fieldDbName (@{$_[0]->{_fieldDbNames}}) {
    #$outAccum->[$idx][$alleleIdx][$positionIdx] = $href->[$self->{_dbName}][$self->{_fieldDbNames}[$idx]] }
    $_[7]->[$idx][$_[5]][$_[6]] = $geneDb->[$fieldDbName];
    $idx++;
  }

  # if($_[0]->{_dist}) {

  # }

  return $_[7];
};

__PACKAGE__->meta->make_immutable;

1;