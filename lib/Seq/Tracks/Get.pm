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

  $self->headers->addFeaturesToHeader([$self->allFeatureNames], $self->name);

  $self->{_fieldDbNames} = [map { $self->getFieldDbName($_) } $self->allFeatureNames];
  $self->{_fieldIdxRange} = [ 0 .. $#{$self->{_fieldDbNames}} ];
}

# Take a hash (that is passed to this function), and get back all features
# that belong to thie Track
# @param <Seq::Tracks::Any> $self
# @param <HashRef> $href : The raw data (presumably from the database);
# @return <HashRef> : A hash ref of featureName => featureValue pairs for
# all features the user specified for this Track in their config file
sub get {
  #my ($self, $href, $chr, $refBase, $altAlleles, $outAccum, $alleleNumber) = @_
  #$href is the data that the user previously grabbed from the database
  # $_[0] == $self
  # $_[1] == $href
  # $_[2] == $chr
  # $_[3] == $refBase
  # $_[4] == $altAlleles
  # $_[5] == $alleleIdx
  # $_[6] == $positionIdx
  # $_[7] == $outAccum
  
  #internally the data is store keyed on the dbName not name, to save space
  # 'some dbName' => someData
  #dbName is simply the track name as stored in the database

  #some features simply don't have any features, and for those just return
  #the value they stored
  if($_[0]->{_noFeatures}) {
    #$outAccum->[$alleleIdx][$positionIdx] = $href->[ $self->{_dbName} ]
    $_[7]->[$_[5]][$_[6]] = $_[1]->[ $_[0]->{_dbName} ];

    #      $outAccum;
    return $_[7];
  }

  # We have features, so let's find those and return them
  # Since all features are stored in some shortened form in the db, we also
  # will first need to get their dbNames ($self->getFieldDbName)
  # and these dbNames will be found as a value of $href->{$self->dbName}
  # #http://ideone.com/WD3Ele
  # return [ map { $_[1]->[$_[0]->{_dbName}][$_] } @{$_[0]->{_fieldDbNames}} ];
  for my $idx (@{$_[0]->{_fieldIdxRange}}) {
    #$outAccum->[$idx][$alleleIdx][$positionIdx] = $href->[$self->{_dbName}][$self->{_fieldDbNames}[$idx]] }
    $_[7]->[$idx][$_[5]][$_[6]] = $_[1]->[$_[0]->{_dbName}][$_[0]->{_fieldDbNames}[$idx]];
  }
        #$outAccum;
  return $_[7];
}
__PACKAGE__->meta->make_immutable;

1;