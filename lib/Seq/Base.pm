use 5.10.0;
use strict;
use warnings;
our $VERSION = '0.002';

package Seq::Base;

# ABSTRACT: Configures singleton log and sets database directory

# Also exports db object, since we configure the database class anyway

# VERSION

use Mouse 2;
use namespace::autoclean;
use Seq::DBManager;
use Seq::Tracks;

use DDP;
#exports new_with_config
with 'Seq::Role::ConfigFromFile', 
#setLogLevel, setLogPath, setPublisher
'Seq::Role::Message',
############# Required Arguments ###########
has database_dir => (is => 'ro', required => 1);

has tracks => (is => 'ro', required => 1);

############ Public Exports ###################
has gettersOnly => (is => 'ro', default => 0);

has tracksObj => (is => 'ro', init_arg => undef, lazy => 1, default => sub {
  my $self = shift;
  return Seq::Tracks->new({tracks => $self->tracks, gettersOnly => $self->gettersOnly });
});

############# Optional Arguments #############
has publisher => (is => 'ro');

has logPath => (is => 'ro');

has verbose => (is => 'ro');

has debug => (is => 'ro', default => 0);

sub BUILDARGS {
  my ($self, $data) = @_;

  # DBManager acts as a singleton. It is configured once, and then consumed repeatedly
  # However, in long running processes, this can lead to misconfiguration issues
  # and worse, environments created in one process, then copied during forking, to others
  # To combat this, every time Seq::Base is called, we re-set/initialzied the static
  # properties that create this behavior
  # Initialize it before BUILD, to make this class less dependent on inheritance order
  Seq::DBManager::initialize();

  # Since we never have more than one database_dir, it's a global property we can set
  # in this package, which Seq.pm and Seq::Build extend from
  Seq::DBManager::setGlobalDatabaseDir($data->{database_dir});

  # Similarly Seq::Role::Message acts as a singleton
  # Clear previous consumer's state, if in long-running process
  Seq::Role::Message::initialize();

  # Each track getter adds its own features to Seq::Headers, which is a singleton
  # Since instantiating Seq::Tracks also instantiates getters at this point
  # We must clear Seq::Headers here to ensure our tracks can properly do this
  # TODO: Make Seq::Headers idempotent, such that one track cannot add its own
  # headers multiple times
  Seq::Headers::initialize();

  # Not really needed for Seq::Tracks, but for clarity
  Seq::Tracks::initialize();

  return $data;
}

sub BUILD {
  my $self = shift;

  # Seq::Role::Message settings
  # We manually set the publisher, logPath, verbosity, and debug, because
  # Seq::Role::Message is meant to be consumed globally, but configured once
  # Treating publisher, logPath, verbose, debug as instance variables
  # would result in having to configure this class in every consuming class
  if(defined $self->publisher) {
    $self->setPublisher($self->publisher);
  }

  if(defined $self->logPath) {
    $self->setLogPath($self->logPath);
  }

  if(defined $self->verbose) {
    $self->setVerbosity($self->verbose);
  }

  #todo: finisih ;for now we have only one level
  if ($self->debug) {
    $self->setLogLevel('DEBUG');
  } else {
    $self->setLogLevel('INFO');
  }

}

__PACKAGE__->meta->make_immutable;

1;
