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
use Carp qw/croak/;

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
  $debug       = 0;
  $verbose     = 10000;
  $publisher   = undef;
  $messageBase = undef;
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
      connect_timeout => 1,
    }
  );

  $messageBase = $publisherConfig->{messageBase};
}

# note, accessing hash directly because traits don't work with Maybe types
sub publishMessage {

  # my ( $self, $msg ) = @_;
  # to save on perf, $_[0] == $self, $_[1] == $msg;

  # because predicates don't trigger builders, need to check hasPublisherAddress
  return unless $publisher;

  $messageBase->{data} = $_[1];

  $publisher->put(
    {
      priority => 0,
      data     => encode_json($messageBase),
    }
  );

  return;
}

sub publishProgress {

  # my ( $self, $annotatedCount, $skippedCount ) = @_;
  #     $_[0],  $_[1],           $_[2]

  # because predicates don't trigger builders, need to check hasPublisherAddress
  return unless $publisher;

  $messageBase->{data} = { progress => $_[1], skipped => $_[2] };

  $publisher->put(
    {
      priority => 0,
      data     => encode_json($messageBase),
    }
  );
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

    # do not publish debug messages by default
    # if($debug) {
    #   $_[0]->publishMessage( "Debug: $_[2]" );
    # }
  }
  elsif ( $_[1] eq 'warn' ) {
    $Seq::Role::Message::LOG->WARN("[$_[1]] $_[2]");

    # we may publish too many warnings this way
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
