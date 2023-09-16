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

# Note that Seq::Headeres is a singleton class
# The headers property is exported only to allow easier overriding of setHeaders
has headers => (
  is       => 'ro',
  init_arg => undef,
  lazy     => 1,
  default  => sub { Seq::Headers->new() },
);

sub BUILD {
  my $self = shift;

  # Skip accesor penalty, the get function in this package may be called
  # hundreds of millions of times
  $self->{_dbName} = $self->dbName;

  #register all features for this track
  #@params $parent, $child
  #if this class has no features, then the track's name is also its only feature
  if ( $self->noFeatures ) {
    return;
  }

  $self->{_fDb} =
    [ map { $self->getFieldDbName($_) } @{ $self->features } ];
  $self->{_fIdx} = [ 0 .. $#{ $self->features } ];
}

# Decouple from build to allow decoupling from dbName / build order
sub setHeaders {
  my $self = shift;

  if ( $self->noFeatures ) {
    $self->headers->addFeaturesToHeader( $self->name );
    return;
  }

  $self->headers->addFeaturesToHeader( $self->features, $self->name );
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
  # $_[3] == <String> $refBase : a base, one of ACTG
  # $_[4] == <String> $allele  : a base, the allele (ACTG or a deletion in the form of "-{number}", or insertion in the form of "+" followed by a sequence of nucleotides)
  # $_[5] == <Int> $posIdx : the position in the indel, if any
  # $_[6] == <ArrayRef> $outAccum : a reference to the output, which we mutate

  #internally the data is store keyed on the dbName not name, to save space
  # 'some dbName' => someData
  #dbName is simply the track name as stored in the database

  #some features simply don't have any features, and for those just return
  #the value they stored
  if ( !$_[0]->{_fIdx} ) {

    #$outAccum->[$posIdx] = $href->[ $self->{_dbName} ]
    # $_[6]->[$_[5]] = $_[1]->[ $_[0]->{_dbName} ];

    #return #$outAccum;
    return $_[6];
  }

  # TODO: decide whether we want to revert to old system of returning a bunch of !
  # one for each feature
  # This is clunky, to have _i and fieldDbNames
  if ( !defined $_[1]->[ $_[0]->{_dbName} ] ) {
    for my $i ( @{ $_[0]->{_fIdx} } ) {
      $_[6]->[$i][ $_[5] ] = undef;
    }

    return $_[6];
  }

  # We have features, so let's find those and return them
  # Since all features are stored in some shortened form in the db, we also
  # will first need to get their dbNames ($self->getFieldDbName)
  # and these dbNames will be found as a value of $href->{$self->dbName}
  # #http://ideone.com/WD3Ele
  # return [ map { $_[1]->[$_[0]->{_dbName}][$_] } @{$_[0]->{_fieldDbNames}} ];
  my $idx = 0;
  for my $fieldDbName ( @{ $_[0]->{_fDb} } ) {

    #$outAccum->[$idx][$posIdx] = $href->[$self->{_dbName}][$fieldDbName] }
    $_[6]->[$idx][ $_[5] ] = $_[1]->[ $_[0]->{_dbName} ][$fieldDbName];
    $idx++;
  }

  #return #$outAccum;
  return $_[6];
}
__PACKAGE__->meta->make_immutable;

1;
