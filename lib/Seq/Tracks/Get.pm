use 5.10.0;
use strict;
use warnings;

package Seq::Tracks::Get;
# Synopsis: For fetching data
# TODO: make this a role?
our $VERSION = '0.001';

use Mouse 2;
use DDP;
extends 'Seq::Tracks::Base';

use Seq::Headers;

has headers => (
  is => 'ro',
  init_arg => undef,
  lazy => 1,
  default => sub { Seq::Headers->new() },
);

sub BUILD {
  my $self = shift;

  # Skip accesor penalty, the get function in this package may be called
  # hundreds of millions of times
  $self->{_dbName} = $self->dbName;

  #register all features for this track
  #@params $parent, $child
  #if this class has no features, then the track's name is also its only feature
  if($self->noFeatures) {
    $self->{_noFeatures} = 1;
    $self->headers->addFeaturesToHeader($self->name);
    return;
  }

  $self->headers->addFeaturesToHeader($self->features, $self->name);

  $self->{_fieldDbNames} = [map { $self->getFieldDbName($_) } @{$self->features}];
}

# Take an array reference containing  (that is passed to this function), and get back all features
# that belong to thie Track
# @param <Seq::Tracks::Any> $self
# @param <HashRef> $href : The raw data (presumably from the database);
# @return <HashRef> : A hash ref of featureName => featureValue pairs for
# all features the user specified for this Track in their config file
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
  
  #internally the data is store keyed on the dbName not name, to save space
  # 'some dbName' => someData
  #dbName is simply the track name as stored in the database

  #some features simply don't have any features, and for those just return
  #the value they stored
  if($_[0]->{_noFeatures}) {
    #$outAccum->[$alleleIdx][$positionIdx] = $href->[ $self->{_dbName} ]
    $_[7]->[$_[5]][$_[6]] = $_[1]->[ $_[0]->{_dbName} ];

    #return #$outAccum;
    return $_[7];
  }

  # We have features, so let's find those and return them
  # Since all features are stored in some shortened form in the db, we also
  # will first need to get their dbNames ($self->getFieldDbName)
  # and these dbNames will be found as a value of $href->{$self->dbName}
  # #http://ideone.com/WD3Ele
  # return [ map { $_[1]->[$_[0]->{_dbName}][$_] } @{$_[0]->{_fieldDbNames}} ];
  my $idx = 0;
  for my $fieldDbName (@{$_[0]->{_fieldDbNames}}) {
    #$outAccum->[$idx][$alleleIdx][$positionIdx] = $href->[$self->{_dbName}][$self->{_fieldDbNames}[$idx]] }
    $_[7]->[$idx][$_[5]][$_[6]] = $_[1]->[$_[0]->{_dbName}][$fieldDbName];
    $idx++;
  }

  #return #$outAccum;
  return $_[7];
}
__PACKAGE__->meta->make_immutable;

1;