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

use DDP;
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
has dbName => ( is => 'ro', init_arg => undef, writer => '_setDbName');

# Some tracks may have a nearest property; these are stored as their own track, but
# conceptually are a sub-track, 
has nearestTrackName => ( is => 'ro', isa => 'Str', init_arg => undef, default => 'nearest');

has nearestDbName => ( is => 'ro', isa => 'Str', init_arg => undef, writer => '_setNearestDbName');

has joinTrackFeatures => (is => 'ro', isa => 'ArrayRef', init_arg => undef, writer => '_setJoinTrackFeatures');

has joinTrackName => (is => 'ro', isa => 'Str', init_arg => undef, writer => '_setJoinTrackName');

###################### Required Arguments ############################
# the track name
has name => ( is => 'ro', isa => 'Str', required => 1);

has type => ( is => 'ro', isa => 'TrackType', required => 1);

has assembly => ( is => 'ro', isa => 'Str', required => 1);
#anything with an underscore comes from the config format
#anything config keys that can be set in YAML but that only need to be used
#during building should be defined here
has chromosomes => (
  is => 'ro',
  isa => 'HashRef',
  traits => ['Hash'],
  handles => {
    allWantedChrs => 'keys',
    chrIsWanted => 'exists',
  },
  required => 1,
);

#http://ideone.com/zfZBO2
# Get the wanted chromosome, or a common transformation if not wanted
# Since Bystro by default expects to use UCSC's naming convention, this may be useful
# Chr may not be flagged as wanted, but actually be wanted for several reasons
# 1) not prepended with 'chr'
# 2) uses NCBI MT instead of chrM
sub normalizedWantedChr {
  #my ($self, $chr) = @_;
  #    $_[0], $_[1]

  #return $self->chromosomes->{$chr} ||
  return $_[0]->chromosomes->{$_[1]} ||
    #($chr eq 'chrMT' ? $self->chromosomes->{'chrM'} : undef) ||
    ($_[1] eq 'chrMT' ? $_[0]->chromosomes->{'chrM'} : undef) ||
    #(index($chr, 'chr') == -1 ? ($chr eq 'MT' ? $self->chromosomes->{'chrM'} : $self->chromosomes->{'chr' . $chr}) : undef);
    (index($_[1], 'chr') == -1 ? ($_[1] eq 'MT' ? $_[0]->chromosomes->{'chrM'} : $_[0]->chromosomes->{'chr' . $_[1]}) : undef);
}

has fieldNames => (is => 'ro', init_arg => undef, default => sub {
  my $self = shift;
  return Seq::Tracks::Base::MapFieldNames->new({name => $self->name,
    assembly => $self->assembly, debug => $self->debug});
}, handles => ['getFieldDbName', 'getFieldName']);

################# Optional arguments ####################
has wantedChr => (is => 'ro', isa => 'Maybe[Str]', lazy => 1, default => undef);

# Using lazy here lets us avoid memory penalties of initializing 
# The features defined in the config file, not all tracks need features
# We allow people to set a feature type for each feature #- feature : int
# We store feature types separately since those are optional as well
# Cannot use predicate with this, because it ALWAYS has a default non-empty value
# As required by the 'Array' trait
has features => (
  is => 'ro',
  isa => 'ArrayRef',
  lazy => 1,
  traits   => ['Array'],
  default  => sub{ [] },
  handles  => { 
    allFeatureNames => 'elements',
    noFeatures  => 'is_empty',
  },
);

# Public, but not expected to be set by calling class, derived from features
# in BUILDARG
has featureDataTypes => (
  is => 'ro',
  isa => 'HashRef[DataType]',
  lazy => 1,
  traits   => ['Hash'],
  default  => sub{{}},
  handles  => { 
    getFeatureType => 'get',
  },
);

has fieldMap => (is => 'ro', isa => 'HashRef', lazy => 1, default => sub{ {} });
# We allow a "nearest" property to be defined for any tracks
# Although it won't make sense for some (like reference)
# It's up to the consuming class to decide if they need it
# It is a property that, when set, may have 0 or more features
# Cannot use predicate with this, because it ALWAYS has a default non-empty value
# As required by the 'Array' trait
has nearest => (
  is => 'ro',
  isa => 'ArrayRef',
  # Cannot use traits with Maybe
  traits => ['Array'],
  handles => {
    noNearestFeatures => 'is_empty',
    allNearestFeatureNames => 'elements',
  },
  lazy => 1,
  default => sub{ [] },
);

has join => (is => 'ro', isa => 'Maybe[HashRef]', predicate => 'hasJoin', lazy => 1, default => undef);

has debug => ( is => 'ro', isa => 'Bool', lazy => 1, default => 0 );

# has index => (is => 'ro', init_arg => undef, default => sub { ++indexOfThisTrack; });
#### Initialize / make dbnames for features and tracks before forking occurs ###
sub BUILD {
  my $self = shift;

  # say "index is";
  # p $self->index;
  # getFieldDbNames is not a pure function; sideEffect of setting auto-generated dbNames in the
  # database the first time (ever) that it is run for a track
  # We could change this effect; for now, initialize here so that each thread
  # gets the same name
  for my $featureName ($self->allFeatureNames) {
    $self->getFieldDbName($featureName);
  }

  my $trackNameMapper = Seq::Tracks::Base::MapTrackNames->new();
  #Set the nearest track names first, because they may be applied genome wide
  #And if we use array format to store data (to save space) good to have
  #Genome-wide tracks have lower indexes, so that higher indexes can be used for 
  #sparser items, because we cannot store a sparse array, must store 1 byte per field
  if(!$self->noNearestFeatures) {
    my $nearestFullQualifiedTrackName = $self->name . '.' . $self->nearestTrackName;

    $self->_setNearestDbName( $trackNameMapper->getOrMakeDbName($nearestFullQualifiedTrackName) );

    $self->log('debug', "Track " . $self->name . ' nearest dbName is ' . $self->nearestDbName);
  }

  $self->_setDbName( $trackNameMapper->getOrMakeDbName($self->name) );

  $self->log('debug', "Track " . $self->name . " dbName is " . $self->dbName);

  if($self->hasJoin) {
    if(!defined $self->join->{track}) {
      $self->log('fatal', "'join' requires track key");
    }

    $self->_setJoinTrackName($self->join->{track});
    $self->_setJoinTrackFeatures($self->join->{features});

    #Each track gets its own private naming of join features
    #Since the track may choose to store these features as arrays
    #Again, needs to happen outside of thread, first time it's ever called
    if($self->joinTrackFeatures) {
      for my $feature (@{$self->joinTrackFeatures}) {
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
sub BUILDARGS {
  my ($class, $data) = @_;

  # #don't mutate the input data
  # my %data = %$dataHref;

  if(defined $data->{chromosomes} &&  ref $data->{chromosomes} eq 'ARRAY') {
    my %chromosomes = map { $_ => $_ } @{$data->{chromosomes} };
    $data->{chromosomes} = \%chromosomes;
  }

  if(defined $data->{wantedChr} ) {
    my @chrs = split(',', $data->{wantedChr});

    my $wantedChrs = {};
    for my $chr (@chrs) {
      if (exists $data->{chromosomes}->{$chr} ) {
        $wantedChrs->{$chr} = $chr;
      } else {
        $class->log('fatal', "Wanted chromosome $chr not listed in chromosomes in YAML config");
      }
    }

    $data->{chromosomes} = $wantedChrs;
  }

  if(! defined $data->{features} ) {
    return $data;
  }

  if( defined $data->{features} && ref $data->{features} ne 'ARRAY') {
    #This actually works :)
    $class->log('fatal', 'features must be array');
  }

  # If features are passed to as hashes (to accomodate their data type) get back to array
  my @featureLabels;

  # The user can rename any input field, this will be used for the feature name
  # This makes it possible to store any name in the db, output file, in place
  # of the field name in the source file used to make the db
  my $fieldMap = $data->{fieldMap} || {};

  for my $origFeature (@{$data->{features} } ) {
    if (ref $origFeature eq 'HASH') {
      my ($name, $type) = %$origFeature; #Thomas Wingo method

      # Transform the feature name if needed
      $name = $fieldMap->{$name} || $name;

      push @featureLabels, $name;
      $data->{featureDataTypes}{$name} = $type;

      next;
    }

    my $name = $fieldMap->{$origFeature} || $origFeature;

    push @featureLabels, $name;
  }

  $data->{features} = \@featureLabels;

  # Currently requires features. Could allow for tracks w/o features in future
  if( defined $data->{nearest} ) {
    if( ref $data->{nearest} ne 'ARRAY' || !@{ $data->{nearest} } ) {
      $class->log('fatal', 'Cannot set "nearest" property without providing 
       an array of feature names');
    }

    for my $nearestFeatureName ( @{$data->{nearest}} ) {
      #~ takes a -1 and makes it a 0
      if(!defined( first{$_ eq $nearestFeatureName} @{$data->{features}} )) {
        $class->log('fatal', "$nearestFeatureName, which you've defined under 
          the nearest property, doesn't exist in the list of $data->{name} 'feature' 
          properties");
      }
    }
  }

  return $data;
};

__PACKAGE__->meta->make_immutable;

1;
