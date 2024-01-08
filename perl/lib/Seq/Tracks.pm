# ABSTRACT: A base class for track classes

# Used to simplify process of detecting tracks
# Tracks.pm knows very little about each track, just enough to instantiate them
# This is a singleton class; it will not instantiate multiple of each track
# Finally, it is worth noting that track order matters
# By default, the tracks are stord in the database in the order specified in the
# tracks object (from the YAML config)
# Since the database stores each track's data at each position in the genome
# in an array, and arrays can only be consequtively indexed, it is best
# to place sparse tracks at the end of the array of tracks
# such that undef values are not written for these sparse tracks for as many positions
# Ex: If we place refSeq track before cadd track, at every position where cadd exists
# and refSeq doesn't, an undef value will be written, such the cadd value can be written
# to the appropriate place;
# ie: [refSeq, cadd] yields [undef, caddData] at every position that cadd exists
# but refSeq doesn't
# However, placing refSeq after cadd maens that we can simply store [cadd] and notice
# that there is no refSeq index

package Seq::Tracks;
use 5.10.0;
use strict;
use warnings;

use Clone 'clone';

use Mouse 2;
with 'Seq::Role::Message';

use Seq::Tracks::Reference;
use Seq::Tracks::Score;
use Seq::Tracks::Sparse;
use Seq::Tracks::Region;
use Seq::Tracks::Gene;
use Seq::Tracks::Cadd;
use Seq::Tracks::Vcf;
use Seq::Tracks::Nearest;

use Seq::Tracks::Reference::Build;
use Seq::Tracks::Score::Build;
use Seq::Tracks::Sparse::Build;
use Seq::Tracks::Region::Build;
use Seq::Tracks::Gene::Build;
use Seq::Tracks::Nearest::Build;
use Seq::Tracks::Cadd::Build;
use Seq::Tracks::Vcf::Build;

use Seq::Tracks::Base::Types;
########################### Configuration ##################################
# This only matters the first time this class is called
# All other calls will ignore this property
has gettersOnly => ( is => 'ro', isa => 'Bool', default => 0 );

# @param <ArrayRef> tracks: track configuration
# expects: {
# typeName : {
#  name: someName (optional),
#  data: {
#   feature1:
#} } }
# This is used to check whether this package has been initialized
has tracks => (
  is  => 'ro',
  isa => 'ArrayRef[HashRef]'
);

has outputOrder => (
  is  => 'ro',
  isa => 'Maybe[ArrayRef[Str]]'
);
########################### Public Methods #################################

# @param <ArrayRef> trackBuilders : ordered track builders
state $orderedTrackBuildersAref = [];
has trackBuilders => (
  is       => 'ro',
  isa      => 'ArrayRef',
  init_arg => undef,
  lazy     => 1,
  traits   => ['Array'],
  handles  => { allTrackBuilders => 'elements' },
  default  => sub { $orderedTrackBuildersAref }
);

state $trackBuildersByName = {};

sub getTrackBuilderByName {
  # my ($self, $name) = @_; #$_[1] == $name
  return $trackBuildersByName->{ $_[1] };
}

state $trackBuildersByType = {};

sub getTrackBuildersByType {
  #my ($self, $type) = @_; #$_[1] == $type
  return $trackBuildersByType->{ $_[1] };
}

# @param <ArrayRef> trackGetters : ordered track getters
state $orderedTrackGettersAref = [];
has trackGetters => (
  is       => 'ro',
  isa      => 'ArrayRef',
  init_arg => undef,
  lazy     => 1,
  traits   => ['Array'],
  handles  => { allTrackGetters => 'elements' },
  default  => sub { $orderedTrackGettersAref }
);

state $trackGettersByName = {};

sub getTrackGetterByName {
  #my ($self, $name) = @_; #$_[1] == $name
  return $trackGettersByName->{ $_[1] };
}

state $trackGettersByType = {};

sub getTrackGettersByType {
  # my ($self, $type) = @_; # $_[1] == $type
  return $trackGettersByType->{ $_[1] };
}

################### Individual track getters ##################

my $types = Seq::Tracks::Base::Types->new();

#Returns 1st reference track
sub getRefTrackGetter {
  my $self = shift;
  return $trackGettersByType->{ $types->refType }[0];
}

sub getTrackGettersExceptReference {
  my $self = shift;

  my @trackGettersExceptReference;
  for my $trackGetter ( @{ $self->trackGetters } ) {
    if ( $trackGetter->type ne $types->refType ) {
      push @trackGettersExceptReference, $trackGetter;
    }
  }

  return \@trackGettersExceptReference;
}

sub allRegionTrackBuilders {
  my $self = shift;
  return $trackBuildersByType->{ $types->regionType };
}

sub allScoreTrackBuilders {
  my $self = shift;
  return $trackBuildersByType->{ $types->scoreType };
}

sub allSparseTrackBuilders {
  my $self = shift;
  return $trackBuildersByType->{ $types->sparseType };
}

sub allGeneTrackBuilders {
  my $self = shift;
  return $trackBuildersByType->{ $types->geneType };
}

#only one ref track allowed, so we return the first
sub getRefTrackBuilder {
  my $self = shift;
  return $trackBuildersByType->{ $types->refType }[0];
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

  if ( !$self->tracks ) {
    if ( !_hasTrackGetters() ) {
      $self->log( 'fatal',
        'First time Seq::Tracks is run tracks configuration must be passed' );
      return;
    }

    #If we do have trackGetters, this necessarily means we've run this builder before
    #So just return, since Seq::Tracks is properly configured
    return;
  }

  # If we're only requesting
  if ( $self->gettersOnly ) {
    $self->_buildTrackGetters( $self->tracks );
    return;
  }

  # If both getters and builders requested, don't mutate the tracks object
  # so that builders get their own distinct configuration
  my $getTracks = clone( $self->tracks );

  # TODO: Lazy, or side-effect free initialization?
  # Builders may have side effects; they may configure
  # the db features available, including any private features
  # If so, create those first

  $self->_buildTrackBuilders( $self->tracks );

  # This ordering necessarily means that Builders cannot have getters in their
  # BUILD step / initialization
  # Need to wait for this to run, after BUILD
  $self->_buildTrackGetters($getTracks);
}

################### Private builders #####################
sub _clearStaticGetters {
  $trackGettersByName      = {};
  $orderedTrackGettersAref = [];
  $trackGettersByType      = {};
}

sub _clearStaticBuilders {
  $trackBuildersByName      = {};
  $orderedTrackBuildersAref = [];
  $trackBuildersByType      = {};
}

sub _hasTrackGetters {
  return
       %{$trackGettersByName}
    && @{$orderedTrackGettersAref}
    && %{$trackGettersByType};
}

sub _hasTrackBuilders {
  return
       %{$trackBuildersByName}
    && @{$orderedTrackBuildersAref}
    && %{$trackBuildersByType};
}

sub _buildTrackGetters {
  my $self                   = shift;
  my $trackConfigurationAref = shift;

  if ( !$trackConfigurationAref ) {
    $self->log( 'fatal', '_buildTrackGetters requires trackConfiguration object' );
  }

  my %seenTrackNames;
  my $seenRef = 0;
  # We may have previously configured this class in a long running process
  # If so, remove the tracks, free the memory
  _clearStaticGetters();

  my %trackOrder;
  if ( defined $self->outputOrder ) {
    my %tracks = map { $_->{name} => $_ } @$trackConfigurationAref;
    my $i      = 0;
    for my $name ( @{ $self->outputOrder } ) {
      if ( !defined $tracks{$name} ) {
        $self->log( 'fatal', "Uknown track $name specified in `outputOrder`" );
      }
      elsif ( $tracks{$name}{no_build} ) {
        $self->log( 'fatal',
          "Track $name specified in `outputOrder` has `no_build` set, which means this track cannot be built, and is likely used only as a 'join' track, joined onto another track."
        );
      }

      $trackOrder{$name} = $i;
      $i++;
    }

    if ( $i < @$trackConfigurationAref ) {
      my @notSeen =
        map { exists $trackOrder{ $_->{name} } || $_->{no_build} ? () : $_->{name} }
        @$trackConfigurationAref;

      if ( @notSeen > 0 ) {
        $self->log( 'fatal',
          "When using `outputOrder`, specify all tracks, unless they have `no_build: true`, missing: "
            . join( ',', @notSeen ) );
      }
    }
  }

  # Iterate over the original order
  # This is important, because otherwise we may accidentally set the
  # tracks database order based on the output order
  # if _buildTrackGetters is called before _buildTrackBuilders
  my $i = 0;
  for my $trackHref (@$trackConfigurationAref) {
    if ( $trackHref->{ref} ) {
      $trackHref->{ref} = $trackBuildersByName->{ $trackHref->{ref} };
    }

    if ( !$seenRef ) {
      $seenRef = $trackHref->{type} eq $types->refType;

      if ( $seenRef && $trackHref->{no_build} ) {
        $self->log( 'fatal', "Reference track cannot have `no_build` set" );
      }
    }
    elsif ( $trackHref->{type} eq $types->refType ) {
      $self->log( 'fatal', "Only one reference track allowed, found at least 2" );
    }

    # If we don't build the track, we also can't fetch data from the track
    # In the rest of the body of this loop we define the track getters
    if ( $trackHref->{no_build} ) {
      next;
    }

    my $className = $self->_toTrackGetterClass( $trackHref->{type} );

    my $track = $className->new($trackHref);

    if ( exists $seenTrackNames{ $track->{name} } ) {
      $self->log(
        'fatal', "More than one track with the same name
        exists: $trackHref->{name}. Each track name must be unique
      . Overriding the last object for this name, with the new"
      );
    }

    #we use the track name rather than the trackHref name
    #because at the moment, users are allowed to rename their tracks
    #by name :
    #   something : someOtherName
    $trackGettersByName->{ $track->{name} } = $track;

    #allows us to preserve order when iterating over all track getters
    if ( %trackOrder && $track->{name} ) {
      $orderedTrackGettersAref->[ $trackOrder{ $track->{name} } ] = $track;
    }
    else {
      $orderedTrackGettersAref->[$i] = $track;
    }

    $i++;
  }

  for my $track (@$orderedTrackGettersAref) {
    $track->setHeaders();
    push @{ $trackGettersByType->{ $track->type } }, $track;
  }

  if ( !$seenRef ) {
    $self->log( 'fatal', "One reference track required, found none" );
  }
}

#different from Seq::Tracks in that we store class instances hashed on track type
#this is to allow us to more easily build tracks of one type in a certain order
sub _buildTrackBuilders {
  my $self                   = shift;
  my $trackConfigurationAref = shift;

  if ( !$trackConfigurationAref ) {
    $self->log( 'fatal', '_buildTrackBuilders requires trackConfiguration object' );
  }

  my %seenTrackNames;
  my $seenRef;
  # We may have previously configured this class in a long running process
  # If so, remove the tracks, free the memory
  _clearStaticBuilders();

  for my $trackHref (@$trackConfigurationAref) {
    if ( $trackHref->{ref} ) {
      $trackHref->{ref} = $trackBuildersByName->{ $trackHref->{ref} };
    }

    my $className = $self->_toTrackBuilderClass( $trackHref->{type} );

    my $track = $className->new($trackHref);

    if ( !$seenRef ) {
      $seenRef = $track->{type} eq $types->refType;

      if ( $track->{no_build} ) {
        $self->log( 'fatal', "Reference track cannot have `no_build` set" );
      }
    }
    elsif ( $track->{type} eq $types->refType ) {
      $self->log( 'fatal', "Only one reference track allowed, found at least 2" );
    }

    #we use the track name rather than the trackHref name
    #because at the moment, users are allowed to rename their tracks
    #by name :
    #   something : someOtherName
    if ( exists $seenTrackNames{ $track->{name} } ) {
      $self->log(
        'fatal', "More than one track with the same name
        exists: $trackHref->{name}. Each track name must be unique
      . Overriding the last object for this name, with the new"
      );
    }

    #we use the track name rather than the trackHref name
    #because at the moment, users are allowed to rename their tracks
    #by name :
    #   something : someOtherName
    #TODO: make this go away by automating track name conversion/storing in db
    $trackBuildersByName->{ $track->{name} } = $track;

    push @{$orderedTrackBuildersAref}, $track;

    push @{ $trackBuildersByType->{ $trackHref->{type} } }, $track;
  }

  if ( !$seenRef ) {
    $self->log( 'fatal', "One reference track required, found none" );
  }
}

####### Helper methods for _buildTrackBulders & _buildTrackGetters methods ########

sub _toTitleCase {
  my $self = shift;
  my $name = shift;

  return uc( substr( $name, 0, 1 ) ) . substr( $name, 1, length($name) - 1 );
}

sub _toTrackGetterClass {
  my $self = shift, my $type = shift;

  # TODO: this right now won't pass $self->type TrackType constraints
  if ( $type =~ /\w+\:+\w+/ ) {
    my @types = split /\:+/, $type;
    my $part1 = $self->_toTitleCase( $types[0] );
    my $part2 = $self->_toTitleCase( $types[1] );

    return "Seq::Tracks::" . $part1 . "::" . $part2;
  }

  return "Seq::Tracks::" . $self->_toTitleCase($type);
}

sub _toTrackBuilderClass {
  my $self = shift, my $type = shift;

  # TODO: this right now won't pass $self->type TrackType constraints
  if ( $type =~ /\w+\:+\w+/ ) {
    my @types = split /\:+/, $type;
    my $part1 = $self->_toTitleCase( $types[0] );
    my $part2 = $self->_toTitleCase( $types[1] );

    return "Seq::Tracks::" . $part1 . "::" . $part2 . "::Build";
  }

  return "Seq::Tracks::" . $self->_toTitleCase($type) . "::Build";
}

__PACKAGE__->meta->make_immutable;
1;
