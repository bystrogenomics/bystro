use 5.10.0;
use strict;
use warnings;

package Seq::Tracks::Base;
#Every track class extends this. The attributes detailed within are used
#regardless of whether we're building or annotating
#and nothing else ends up here (for instance, required_fields goes to Tracks::Build)

our $VERSION = '0.001';

use Mouse 2;
use MouseX::NativeTraits;

use List::Util qw/first/;

use Seq::Tracks::Base::MapTrackNames;
use Seq::Tracks::Base::MapFieldNames;
use Seq::DBManager;
# use String::Util qw/trim/;

# automatically imports TrackType
use Seq::Tracks::Base::Types;

with 'Seq::Role::Message';

# state $indexOfThisTrack = 0;
################# Public Exports ##########################
# Not lazy because every track will use this 100%, and rest are trivial values
# Not worth complexity of Maybe[Type], default => undef,
has dbName => ( is => 'ro', init_arg => undef, writer => '_setDbName' );

# Don't actually build, used if we want a track to serve as a join track
# Where the track must exist in the tracks list, so that the "join" property of a different track can reference it
# but we don't want to build it
# An example of this is clinvar. We don't want to build the actual track, because it only allows for position-wise
# matching, rather than allele-wise, but it contains large structural variations which we want to capture
# where they overlap with genes.
# So we join this clinvar data on refSeq data, capturing larger variation, but don't build a clinvar track
# using this data; instead we build a clinvar track using the VCF dataset, and call that clinvarVcf
has no_build => ( is => 'ro', isa => 'Bool', default => 0 );

# TODO: Evaluate removing joinTracks in favor of utilities
# or otherwise make them more flexible (array of them)
has joinTrackFeatures => (
  is       => 'ro',
  isa      => 'ArrayRef',
  init_arg => undef,
  writer   => '_setJoinTrackFeatures'
);

has joinTrackName =>
  ( is => 'ro', isa => 'Str', init_arg => undef, writer => '_setJoinTrackName' );

###################### Required Arguments ############################
# the track name
has name => ( is => 'ro', isa => 'Str', required => 1 );

has type => ( is => 'ro', isa => 'TrackType', required => 1 );

has assembly => ( is => 'ro', isa => 'Str', required => 1 );
#anything with an underscore comes from the config format
#anything config keys that can be set in YAML but that only need to be used
#during building should be defined here
has chromosomes => (
  is      => 'ro',
  isa     => 'HashRef',
  traits  => ['Hash'],
  handles => {
    allWantedChrs => 'keys',
    chrIsWanted   => 'exists',
  },
  required => 1,
);

# Memoized normalizer of chromosome names
# Handles chromosomes with or without 'chr' prefix
# And interconverts between MT and M, such that the requested
# (MT or M) is returned for the other of the pair
# Ex: for "1" we'll accept chr1 or 1, and both will map to "1"
# Ex: for "chrM" we'll accept chrMT, MT, chrM, M, all mapping to "chrM"
has normalizedWantedChr => (
  is       => 'ro',
  isa      => 'HashRef',
  init_arg => undef,
  lazy     => 1,
  default  => sub {
    my $self = shift;

    # Includes the track chromosomes, with and without "chr" prefixes
    # and if MT or M is provided, the other of MT or M
    my %chromosomes = map { $_ => $_ } keys %{ $self->chromosomes };

    # Add if not already present
    if ( $chromosomes{'MT'} ) {
      $chromosomes{'chrM'}  //= 'MT';
      $chromosomes{'chrMT'} //= 'MT';
      $chromosomes{'M'}     //= 'MT';
    }
    elsif ( $chromosomes{'chrMT'} ) {
      $chromosomes{'chrM'} //= 'chrMT';
      $chromosomes{'MT'}   //= 'chrMT';
      $chromosomes{'M'}    //= 'chrMT';
    }
    elsif ( $chromosomes{'chrM'} ) {
      $chromosomes{'MT'}    //= 'chrM';
      $chromosomes{'chrMT'} //= 'chrM';
      $chromosomes{'M'}     //= 'chrM';
    }
    elsif ( $chromosomes{'M'} ) {
      $chromosomes{'MT'}    //= 'M';
      $chromosomes{'chrMT'} //= 'M';
      $chromosomes{'chrM'}  //= 'M';
    }

    # If provide 'chr' prefixes, map the same chromosomes without those prefixes
    # to the 'chr'-prefix name
    # And vice versa
    for my $chr ( keys %chromosomes ) {
      if ( substr( $chr, 0, 3 ) eq 'chr' ) {
        # Add if not already present, in case user for some reason wants to
        # have chr1 and 1 point to distinct databases.
        my $part = substr( $chr, 3 );

        $chromosomes{$part} //= $chr;
      }
      else {
        # Modify only if not already present
        $chromosomes{"chr$chr"} //= $chr;
      }
    }

    return \%chromosomes;
  }
);

has fieldNames => (
  is       => 'ro',
  init_arg => undef,
  default  => sub {
    my $self = shift;
    return Seq::Tracks::Base::MapFieldNames->new(
      {
        name     => $self->name,
        assembly => $self->assembly,
        debug    => $self->debug
      }
    );
  },
  handles => [ 'getFieldDbName', 'getFieldName' ]
);

################# Optional arguments ####################
has wantedChr => ( is => 'ro', isa => 'Maybe[Str]', lazy => 1, default => undef );

# Using lazy here lets us avoid memory penalties of initializing
# The features defined in the config file, not all tracks need features
# We allow people to set a feature type for each feature #- feature : int
# We store feature types separately since those are optional as well
# Cannot use predicate with this, because it ALWAYS has a default non-empty value
# As required by the 'Array' trait
has features => (
  is      => 'ro',
  isa     => 'ArrayRef',
  lazy    => 1,
  traits  => ['Array'],
  default => sub { [] },
  handles => { noFeatures => 'is_empty', },
);

# Public, but not expected to be set by calling class, derived from features
# in BUILDARG
has featureDataTypes => (
  is      => 'ro',
  isa     => 'HashRef[DataType]',
  lazy    => 1,
  traits  => ['Hash'],
  default => sub { {} },
  handles => { getFeatureType => 'get', },
);

has join => (
  is        => 'ro',
  isa       => 'Maybe[HashRef]',
  predicate => 'hasJoin',
  lazy      => 1,
  default   => undef
);

has debug => ( is => 'ro', isa => 'Bool', lazy => 1, default => 0 );

# has index => (is => 'ro', init_arg => undef, default => sub { ++indexOfThisTrack; });
#### Initialize / make dbnames for features and tracks before forking occurs ###
sub BUILD {
  my $self = shift;
  # getFieldDbNames is not a pure function; sideEffect of setting auto-generated dbNames in the
  # database the first time (ever) that it is run for a track
  # We could change this effect; for now, initialize here so that each thread
  # gets the same name
  for my $featureName ( @{ $self->features } ) {
    $self->getFieldDbName($featureName);
  }

  my $trackNameMapper = Seq::Tracks::Base::MapTrackNames->new();

  $self->_setDbName( $trackNameMapper->getOrMakeDbName( $self->name ) );

  $self->log( 'debug', "Track " . $self->name . " dbName is " . $self->dbName );

  if ( $self->hasJoin ) {
    if ( !defined $self->join->{track} ) {
      $self->log( 'fatal', "'join' requires track key" );
    }

    $self->_setJoinTrackName( $self->join->{track} );
    $self->_setJoinTrackFeatures( $self->join->{features} );

    #Each track gets its own private naming of join features
    #Since the track may choose to store these features as arrays
    #Again, needs to happen outside of thread, first time it's ever called
    if ( $self->joinTrackFeatures ) {
      for my $feature ( @{ $self->joinTrackFeatures } ) {
        $self->getFieldDbName($feature);
      }
    }
  }

  # Commit, sync, and remove any databases opened
  # This is useful because locking may occur if there is an open transaction
  # before fork(), and to make sure that any database meta data is properly
  # committed before tracks begin to use that data.
  Seq::DBManager::cleanUp();
}

############ Argument configuration to meet YAML config spec ###################

# Expects a hash, will crash and burn if it doesn't
around BUILDARGS => sub {
  my ( $orig, $class, $data ) = @_;

  # #don't mutate the input data
  # my %data = %$dataHref;
  if ( defined $data->{chromosomes} && ref $data->{chromosomes} eq 'ARRAY' ) {
    my %chromosomes = map { $_ => $_ } @{ $data->{chromosomes} };
    $data->{chromosomes} = \%chromosomes;
  }

  if ( defined $data->{wantedChr} ) {
    my @chrs = split( ',', $data->{wantedChr} );

    my $wantedChrs = {};
    for my $chr (@chrs) {
      if ( exists $data->{chromosomes}->{$chr} ) {
        $wantedChrs->{$chr} = $chr;
      }
      else {
        $class->log( 'fatal',
          "Wanted chromosome $chr not listed in chromosomes in YAML config" );
      }
    }

    $data->{chromosomes} = $wantedChrs;
  }

  if ( !defined $data->{features} ) {
    return $class->$orig($data);
  }

  if ( defined $data->{features} && ref $data->{features} ne 'ARRAY' ) {
    #This actually works :)
    $class->log( 'fatal', 'features must be array' );
  }

  # If features are passed to as hashes (to accomodate their data type) get back to array
  my @featureLabels;
  my %seenFeatures;
  for my $origFeature ( @{ $data->{features} } ) {
    if ( ref $origFeature eq 'HASH' ) {
      my ( $name, $type ) = %$origFeature; #Thomas Wingo method

      push @featureLabels, $name;
      $data->{featureDataTypes}{$name} = $type;

      next;
    }

    push @featureLabels, $origFeature;
  }

  my $idx = 0;
  for my $feat (@featureLabels) {
    if ( $seenFeatures{$feat} ) {
      $class->log( 'warn',
        "$feat is listed twice under " . $data->{name} . " features, removing" );
      splice( @featureLabels, $idx, 1 );
    }

    $seenFeatures{$feat} = 1;

    $idx++;
  }

  $data->{features} = \@featureLabels;

  return $class->$orig($data);
};

__PACKAGE__->meta->make_immutable;

1;
