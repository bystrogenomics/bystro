# ABSTRACT: A base class for track classes

# used to simplify process of detecting tracks
# I think that Tracks.pm should know which features it has access to
# and anything conforming to that interface should become an instance
# of the appropriate class
# and everythign else shouldn't, and should generate a warning
# This is heavily inspired by Dr. Thomas Wingo's primer picking software design
# expects structure to be {
#  trackName : {typeStuff},
#  typeName2 : {typeStuff2},
#}

#We don't instantiate a new object for each data source
#Instead, we simply create a container for each name : type pair
#We could use an array, but a hash is easier to reason about
#We also expect that each record will be identified by its track name
#so (in db) {
#   trackName : {
#     featureName: featureValue  
#} 
#}

package Seq::Tracks;
use 5.10.0;
use strict;
use warnings;
use DDP;

use Mouse 2;
with 'Seq::Role::Message';

use Seq::Headers;

use Seq::Tracks::Reference;
use Seq::Tracks::Score;
use Seq::Tracks::Sparse;
use Seq::Tracks::Region;
use Seq::Tracks::Gene;
use Seq::Tracks::Cadd;

use Seq::Tracks::Reference::Build;
use Seq::Tracks::Score::Build;
use Seq::Tracks::Sparse::Build;
use Seq::Tracks::Region::Build;
use Seq::Tracks::Gene::Build;
use Seq::Tracks::Cadd::Build;

use Seq::Tracks::Base::Types;
########################### Configuration ##################################
# This only matters the first time this class is called
# All other calls will ignore this property
has gettersOnly => (is => 'ro', isa => 'Bool', default => 0);

# @param <ArrayRef> tracks: track configuration
# expects: {
  # typeName : {
  #  name: someName (optional),
  #  data: {
  #   feature1:   
#} } }
# This is used to check whether this package has been initialized
has tracks => (
  is => 'ro',
  isa => 'ArrayRef[HashRef]'
);

########################### Public Methods #################################

# @param <ArrayRef> trackBuilders : ordered track builders
state $orderedTrackBuildersAref = [];
has trackBuilders => ( is => 'ro', isa => 'ArrayRef', init_arg => undef, lazy => 1,
  traits => ['Array'], handles => { allTrackBuilders => 'elements' }, 
  default => sub { $orderedTrackBuildersAref } );

state $trackBuildersByName = {};
sub getTrackBuilderByName {
  # my ($self, $name) = @_; #$_[1] == $name
  return $trackBuildersByName->{$_[1]};
}

state $trackBuildersByType = {};
sub getTrackBuildersByType {
  #my ($self, $type) = @_; #$_[1] == $type
  return $trackBuildersByType->{$_[1]};
}

# @param <ArrayRef> trackGetters : ordered track getters
state $orderedTrackGettersAref = [];
has trackGetters => ( is => 'ro', isa => 'ArrayRef', init_arg => undef, lazy => 1,
  traits => ['Array'], handles => { allTrackGetters => 'elements' } , 
  default => sub { $orderedTrackGettersAref } );

state $trackGettersByName = {};
sub getTrackGetterByName {
  #my ($self, $name) = @_; #$_[1] == $name
  return $trackGettersByName->{$_[1]};
}

state $trackGettersByType = {};
sub getTrackGettersByType {
  # my ($self, $type) = @_; # $_[1] == $type
  return $trackGettersByType->{$_[1]};
}

################### Individual track getters ##################

my $types = Seq::Tracks::Base::Types->new();

#Returns 1st reference track
sub getRefTrackGetter {
  my $self = shift;
  return $trackGettersByType->{$types->refType}[0];
}

sub getTrackGettersExceptReference {
  my $self = shift;

  my @trackGettersExceptReference;
  for my $trackGetter (@{$self->trackGetters}) {
    if($trackGetter->type ne $types->refType) {
      push @trackGettersExceptReference, $trackGetter;
    }
  }

  return \@trackGettersExceptReference;
}

sub allRegionTrackBuilders {
  my $self = shift;
  return $trackBuildersByType->{$types->regionType};
}

sub allScoreTrackBuilders {
  my $self = shift;
  return $trackBuildersByType->{$types->scoreType};
}

sub allSparseTrackBuilders {
  my $self = shift;
  return $trackBuildersByType->{$types->sparseType};
}

sub allGeneTrackBuilders {
  my $self = shift;
  return $trackBuildersByType->{$types->geneType};
}

#only one ref track allowed, so we return the first
sub getRefTrackBuilder {
  my $self = shift;
  return $trackBuildersByType->{$types->refType}[0];
}

# Used solely for clarity, keep with the interface used in other singleton classes
sub initialize {
  _clearStaticGetters();
  _clearStaticBuilders();
}

sub BUILD {
  my $self = shift;

  # The goal of this class is to allow one consumer to configure the tracks
  # for the rest of the program
  # i.e Seq.pm passes { tracks => $someTrackConfiguration } and Seq::Tracks::Gene
  # can call  Seq::Tracks::getRefTrackGetter and receive a configured ref track getter

  # However it is important that in long-running parent processes, which may 
  # instantiate this program more than once, we do not re-use old configurations
  # So every time the parent passes a tracks object, we re-configure this class
  if( !$self->tracks ) {
    if(!_hasTrackGetters() ) {
      $self->log('fatal', 'First time Seq::Tracks is run tracks configuration must be passed');
      return;
    }

    #If we do have trackGetters, this necessarily means we've run this builder before
    #So just return, since Seq::Tracks is properly configured
    return;
  }

  $self->_buildTrackGetters($self->tracks);

  if($self->gettersOnly) {
    return;
  }

  $self->_buildTrackBuilders($self->tracks);
}

################### Private builders #####################
sub _clearStaticGetters {
  $trackGettersByName = {};
  $orderedTrackGettersAref = [];
  $trackGettersByType = {};
}

sub _clearStaticBuilders {
  $trackBuildersByName = {};
  $orderedTrackBuildersAref = [];
  $trackBuildersByType = {};
}

sub _hasTrackGetters {
  return %{$trackGettersByName} && @{$orderedTrackGettersAref} && %{$trackGettersByType};
}

sub _hasTrackBuilders {
  return  %{$trackBuildersByName} && @{$orderedTrackBuildersAref} && %{$trackBuildersByType};
}

sub _buildTrackGetters {
  my $self = shift;
  my $trackConfigurationAref = shift;

  if(!$trackConfigurationAref) {
    $self->log('fatal', '_buildTrackGetters requires trackConfiguration object');
  }

  my %seenTrackNames;
  my $seenRef = 0;
  # We may have previously configured this class in a long running process
  # If so, remove the tracks, free the memory
  _clearStaticGetters();

  for my $trackHref (@$trackConfigurationAref ) {
    #get the trackClass
    my $trackFileName = $self->_toTrackGetterClass($trackHref->{type} );
    #class 
    my $className = $self->_toTrackGetterClass( $trackHref->{type} );

    my $track = $className->new($trackHref);

    if(!$seenRef) {
      $seenRef = $track->{type} eq $types->refType;
    } elsif($track->{type} eq $types->refType) {
      $self->log('fatal', "Only one reference track allowed, found at least 2");
      return;
    }

    if(exists $seenTrackNames{ $track->{name} } ) {
      $self->log('fatal', "More than one track with the same name 
        exists: $trackHref->{name}. Each track name must be unique
      . Overriding the last object for this name, with the new")
    }

    #we use the track name rather than the trackHref name
    #because at the moment, users are allowed to rename their tracks
    #by name : 
      #   something : someOtherName
    $trackGettersByName->{$track->{name} } = $track;

    #allows us to preserve order when iterating over all track getters
    push @$orderedTrackGettersAref, $track;

    push @{$trackGettersByType->{$trackHref->{type} } }, $track;
  }

  if(!$seenRef) {
    $self->log('fatal', "One reference track required, found none");
  }
}

#different from Seq::Tracks in that we store class instances hashed on track type
#this is to allow us to more easily build tracks of one type in a certain order
sub _buildTrackBuilders {
  my $self = shift;
  my $trackConfigurationAref = shift;

  if(!$trackConfigurationAref) {
    $self->log('fatal', '_buildTrackBuilders requires trackConfiguration object');
  }

  my %seenTrackNames;
  my $seenRef;
  # We may have previously configured this class in a long running process
  # If so, remove the tracks, free the memory
  _clearStaticBuilders();

  for my $trackHref (@$trackConfigurationAref) {
    my $trackFileName = $self->_toTrackBuilderClass($trackHref->{type} );
    #class 
    my $className = $self->_toTrackBuilderClass( $trackHref->{type} );

    my $track = $className->new($trackHref);

    if(!$seenRef) {
      $seenRef = $track->{type} eq $types->refType;
    } elsif($track->{type} eq $types->refType){
      $self->log('fatal', "Only one reference track allowed, found at least 2");
      return;
    }

    #we use the track name rather than the trackHref name
    #because at the moment, users are allowed to rename their tracks
    #by name : 
      #   something : someOtherName
    if(exists $seenTrackNames{ $track->{name} } ) {
      $self->log('fatal', "More than one track with the same name 
        exists: $trackHref->{name}. Each track name must be unique
      . Overriding the last object for this name, with the new")
    }

    #we use the track name rather than the trackHref name
    #because at the moment, users are allowed to rename their tracks
    #by name : 
      #   something : someOtherName
    #TODO: make this go away by automating track name conversion/storing in db
    $trackBuildersByName->{$track->{name} } = $track;

    push @{$orderedTrackBuildersAref}, $track;

    push @{$trackBuildersByType->{$trackHref->{type} } }, $track;
  }

  if(!$seenRef) {
    $self->log('fatal', "One reference track required, found none");
  }
}

####### Helper methods for _buildTrackBulders & _buildTrackGetters methods ########

sub _toTitleCase {
  my $self = shift;
  my $name = shift;

  return uc( substr($name, 0, 1) ) . substr($name, 1, length($name) - 1);
}

sub _toTrackGetterClass {
  my $self = shift,
  my $type = shift;

  my $classNamePart = $self->_toTitleCase($type);

  return "Seq::Tracks::" . $classNamePart;
}

sub _toTrackBuilderClass{
  my $self = shift,
  my $type = shift;

  my $classNamePart = $self->_toTitleCase($type);

  return "Seq::Tracks::" . $classNamePart ."::Build";
}

__PACKAGE__->meta->make_immutable;
1;