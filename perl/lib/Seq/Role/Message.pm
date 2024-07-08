package Seq::Role::Message;
use 5.10.0;
use strict;
use warnings;

our $VERSION = '0.001';

# ABSTRACT: A class for communicating to log and to some plugged in messaging service
# VERSION
use Mouse::Role 2;

#doesn't work with Parallel::ForkManager;
#for more on AnyEvent::Log
#http://search.cpan.org/~mlehmann/AnyEvent-7.12/lib/AnyEvent/Log.pm
# use AnyEvent;
# use AnyEvent::Log;

use Log::Fast;
use namespace::autoclean;
use Beanstalk::Client;
use Cpanel::JSON::XS;
use DDP return_value => 'dump';
use Carp        qw/croak/;
use Time::HiRes qw(time);
use Try::Tiny;

my $PUBLISHER_ACTION_TIMEOUT  = 30;
my $PUBLISHER_CONNECT_TIMEOUT = 10;
my $MAX_PUT_MESSAGE_TIMEOUT   = 5;
# How many consecutive failures to connect to the publisher before we stop trying
my $MAX_PUBLISHER_FAILURES_IN_A_ROW = 5;

$Seq::Role::Message::LOG = Log::Fast->new(
  {
    level  => 'INFO',
    prefix => '%D %T ',
    type   => 'fh',
    fh     => \*STDERR,
  }
);

$Seq::Role::Message::mapLevels = {
  info   => 'INFO',  #\&{$LOG->INFO}
  INFO   => 'INFO',
  ERR    => 'ERR',
  error  => 'ERR',
  fatal  => 'ERR',
  warn   => 'WARN',
  WARN   => 'WARN',
  debug  => 'DEBUG',
  DEBUG  => 'DEBUG',
  NOTICE => 'NOTICE',
};

my %mapSeverity = (
  debug => 0,
  info  => 1,
  warn  => 2,
  error => 3,
  fatal => 4,
);

# Static variables; these need to be cleared by the consuming class
state $debug   = 0;
state $verbose = 1000;
state $publisher;
state $messageBase;
state $lastPublisherInteractionTime;
state $publisherConsecutiveConnectionFailures = 0;

# whether log level or verbosity is at the debug level
# shoud only be accessed after setLogLevel and/or setVerbosity executed if program doesn't want default
has hasDebugLevel => (
  is       => 'ro',
  isa      => 'Bool',
  init_arg => undef,
  lazy     => 1,
  default  => sub {
    return $debug || $verbose == 0;
  }
);
# should only be run after setPublisher is executed if program doesn't want default
has hasPublisher => (
  is       => 'ro',
  isa      => 'Bool',
  init_arg => undef,
  lazy     => 1,
  default  => sub {
    return !!$publisher;
  }
);

sub initialize {
  $debug                                  = 0;
  $verbose                                = 10000;
  $publisher                              = undef;
  $lastPublisherInteractionTime           = 0;
  $publisherConsecutiveConnectionFailures = 0;
  $messageBase                            = undef;
}

sub putMessageWithTimeout {
  my ( $publisher, $timeout, @args ) = @_;

  eval {
    local $SIG{ALRM} = sub { die "TIMED_OUT_CLIENT_SIDE" };
    alarm($timeout);

    $publisher->put(@args); # Execute the passed function with arguments

    alarm(0);               # Disable the alarm
  };

  if ($@) {
    return $@;
  }

  return;
}

sub setLogPath {
  my ( $self, $path ) = @_;
  #open($Seq::Role::Message::Fh, '<', $path);

  #Results in deep recursion issue if we include Seq::Role::IO (which requires Role::Message
  open( my $fh, '>', $path ) or die "Couldn't open log path $path";

  #$AnyEvent::Log::LOG->log_to_file ($path);
  $Seq::Role::Message::LOG->config( { fh => $fh, } );
}

sub setLogLevel {
  my ( $self, $level ) = @_;

  our $mapLevels;

  if ( $level =~ /debug/i ) {
    $debug = 1;
  }

  $Seq::Role::Message::LOG->level( $mapLevels->{$level} );
}

sub setVerbosity {
  my ( $self, $verboseLevel ) = @_;

  if ( $verboseLevel != 0 && $verboseLevel != 1 && $verboseLevel != 2 ) {
    # Should log this
    say STDERR "Verbose level must be 0, 1, or 2, setting to 10000 (no verbose output)";
    $verbose = 10000;
    return;
  }

  $verbose = $verboseLevel;
}

sub setPublisher {
  my ( $self, $publisherConfig ) = @_;

  if ( !ref $publisherConfig eq 'Hash' ) {
    return $self->log->( 'fatal', 'setPublisherAndAddress requires hash' );
  }

  if (
    !(
         defined $publisherConfig->{server}
      && defined $publisherConfig->{queue}
      && defined $publisherConfig->{messageBase}
    )
    )
  {
    return $self->log( 'fatal', 'setPublisher server, queue, messageBase properties' );
  }

  $publisher = Beanstalk::Client->new(
    {
      server          => $publisherConfig->{server},
      default_tube    => $publisherConfig->{queue},
      connect_timeout => $PUBLISHER_CONNECT_TIMEOUT,
    }
  );

  $lastPublisherInteractionTime           = time();
  $publisherConsecutiveConnectionFailures = 0;

  $messageBase = $publisherConfig->{messageBase};
}

sub _incrementPublishFailuresAndWarn {
  $publisherConsecutiveConnectionFailures++;
  if ( $publisherConsecutiveConnectionFailures >= $MAX_PUBLISHER_FAILURES_IN_A_ROW ) {
    say STDERR
      "Exceeded maximum number of progress publisher reconnection attempts. Disabling progress publisher until job completion.";
  }
}

# note, accessing hash directly because traits don't work with Maybe types
sub publishMessage {
  # my ( $self, $msg ) = @_;
  # to save on perf, $_[0] == $self, $_[1] == $msg;

  return unless $publisher;

  if ( $publisherConsecutiveConnectionFailures >= $MAX_PUBLISHER_FAILURES_IN_A_ROW ) {
    return;
  }

  my $timeSinceLastInteraction = time() - $lastPublisherInteractionTime;
  if ( $timeSinceLastInteraction >= $PUBLISHER_ACTION_TIMEOUT ) {
    say
      "Attempting to reconnect progress publisher because time since last interaction is $timeSinceLastInteraction seconds.";

    $publisher->disconnect();
    $publisher->connect();

    # Ensure that we space apart reconnection attempts
    $lastPublisherInteractionTime = time();

    if ( $publisher->error ) {
      say STDERR "Failed to connect to progress publisher server: " . $publisher->error;

      _incrementPublishFailuresAndWarn();
      return;
    }

    $publisherConsecutiveConnectionFailures = 0;
  }

  $messageBase->{data} = $_[1];

  my $error = putMessageWithTimeout( $publisher, $MAX_PUT_MESSAGE_TIMEOUT,
    { priority => 0, data => encode_json($messageBase) } );

  if ( $error || $publisher->error ) {
    my $err = $publisher->error ? $publisher->error : $error;
    say STDERR "Failed to publish message: " . $err;

    return;
  }

  $lastPublisherInteractionTime = time();
}

sub publishProgress {
  # my ( $self, $annotatedCount, $skippedCount ) = @_;
  #     $_[0],  $_[1],           $_[2]

  return unless $publisher;

  if ( $publisherConsecutiveConnectionFailures >= $MAX_PUBLISHER_FAILURES_IN_A_ROW ) {
    return;
  }

  my $timeSinceLastInteraction = time() - $lastPublisherInteractionTime;
  if ( $timeSinceLastInteraction >= $PUBLISHER_ACTION_TIMEOUT ) {
    say
      "Attempting to reconnect progress publisher because time since last interaction is $timeSinceLastInteraction seconds.";

    $publisher->disconnect();
    $publisher->connect();

    # Ensure that we space apart reconnection attempts
    $lastPublisherInteractionTime = time();

    if ( $publisher->error ) {
      say STDERR "Failed to connect to progress publisher server: " . $publisher->error;

      _incrementPublishFailuresAndWarn();
      return;
    }

    $publisherConsecutiveConnectionFailures = 0;
  }

  $messageBase->{data} = { progress => $_[1], skipped => $_[2] };

  my $error = putMessageWithTimeout( $publisher, $MAX_PUT_MESSAGE_TIMEOUT,
    { priority => 0, data => encode_json($messageBase) } );
  if ( $error || $publisher->error ) {
    my $err = $publisher->error ? $publisher->error : $error;
    say STDERR "Failed to publish progress: " . $err;

    return;
  }

  $lastPublisherInteractionTime = time();
}

sub log {
  #my ( $self, $log_method, $msg ) = @_;
  #$_[0] == $self, $_[1] == $log_method, $_[2] == $msg;

  if ( ref $_[2] ) {
    $_[2] = p $_[2];
  }

  if ( $_[1] eq 'info' ) {
    $Seq::Role::Message::LOG->INFO("[$_[1]] $_[2]");

    $_[0]->publishMessage( $_[2] );
  }
  elsif ( $_[1] eq 'debug' ) {
    $Seq::Role::Message::LOG->DEBUG("[$_[1]] $_[2]");

    # we may publish too many debug messages. to enable:
    # $_[0]->publishMessage( "Debug: $_[2]" );
  }
  elsif ( $_[1] eq 'warn' ) {
    $Seq::Role::Message::LOG->WARN("[$_[1]] $_[2]");

    # we may publish too many warnings. to enable:
    # $_[0]->publishMessage( "Warning: $_[2]" );
  }
  elsif ( $_[1] eq 'error' ) {
    $Seq::Role::Message::LOG->ERR("[$_[1]] $_[2]");

    $_[0]->publishMessage("Error: $_[2]");
  }
  elsif ( $_[1] eq 'fatal' ) {
    $Seq::Role::Message::LOG->ERR("[$_[1]] $_[2]");

    $_[0]->publishMessage("Fatal: $_[2]");

    croak("[$_[1]] $_[2]");
  }
  else {
    return;
  }

  # So if verbosity is set to 1, only err, warn, and fatal messages
  # will be printed to sdout
  if ( $verbose <= $mapSeverity{ $_[1] } ) {
    say STDERR "[$_[1]] $_[2]";
  }

  return;
}

no Mouse::Role;
1;
