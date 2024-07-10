
#!/usr/bin/env perl
# Name:           snpfile_annotate_mongo_redis_queue.pl
# Description:
# Date Created:   Wed Dec 24
# By:             Alex Kotlar
# Requires: Snpfile::AnnotatorBase

use 5.10.0;

use strict;
use warnings;

use Beanstalk::Client;
use Cpanel::JSON::XS;

use DDP output => *STDOUT;
use File::Basename;
use Getopt::Long;
use Log::Any::Adapter;
use MCE::Shared;
use MCE::Hobo;
use Path::Tiny qw/path/;
use Try::Tiny;
use Time::HiRes qw(time);
use YAML::XS    qw/LoadFile/;

use Seq;

my ( $verbose, $queueConfigPath, $maxThreads );

GetOptions(
  'v|verbose=i'     => \$verbose,
  'q|queueConfig=s' => \$queueConfigPath,
  'maxThreads=i'    => \$maxThreads,
);

if ( !($queueConfigPath) ) {
  # Generate a help strings that shows the arguments
  say "\nUsage: perl $0 -q <queueConfigPath> --maxThreads <maxThreads>\n";
  exit 1;
}

my $type = 'annotation';

my $PROGRESS  = "progress";
my $FAILED    = "failed";
my $STARTED   = "started";
my $COMPLETED = "completed";

my $conf = LoadFile($queueConfigPath);

# The properties that we accept from the worker caller
my %requiredForAll = (
  output_file_base => 'outputBasePath',
  assembly         => 'assembly',
);

my $requiredForType = { input_files => 'inputFilePath' };

my $configPathBaseDir  = "config/";
my $configFilePathHref = {};

my $queueConfig = $conf->{beanstalkd}{tubes}{$type};

if ( !$queueConfig ) {
  die "Queue config format not recognized. Options are "
    . ( join( ', ', @{ keys %{ $conf->{beanstalkd}{tubes} } } ) );
}

say "Running Annotation queue server";

my $BEANSTALKD_RESERVE_TIMEOUT = 20;
# Needs to be larger than RESERVE_TIMEOUT
# Used for both producer and consumer clients
# We give move time for the producer to connect
# Than in Message.pm, because these messages indiciate job success/failure,
# And therefore is more important than progress updates
my $BEANSTALKD_CONNECT_TIMEOUT = 30;
my $BEANSTALKD_JOB_HEARTBEAT   = 30;

sub _getConsumerClient {
  my $address = shift;
  my $tube    = shift;

  return Beanstalk::Client->new(
    {
      server          => $address,
      default_tube    => $tube,
      connect_timeout => $BEANSTALKD_CONNECT_TIMEOUT,
      encoder         => sub { encode_json( \@_ ) },
      decoder         => sub { @{ decode_json(shift) } },
    }
  );
}

sub _getProducerClient {
  my $address = shift;
  my $tube    = shift;

  return Beanstalk::Client->new(
    {
      server          => $address,
      default_tube    => $tube,
      connect_timeout => $BEANSTALKD_CONNECT_TIMEOUT,
      encoder         => sub { encode_json( \@_ ) },
      decoder         => sub { @{ decode_json(shift) } },
    }
  );
}

sub connectJob {
  my $job     = shift;
  my $address = shift;
  my $socket  = $job->client->connect( $address, $BEANSTALKD_CONNECT_TIMEOUT );

  if ( $job->client->error ) {
    say STDERR "Failed to connect to queue server with error " . $job->client->error;
    return $job->client->error;
  }
  elsif ( !$socket ) {
    say STDERR "Failed to connect to queue server for an unknown reason";
    return "UNKNOWN_CONNECT_ERROR";
  }
  return
}

# Function to execute Beanstalkd command with timeout
sub executeWithTimeout {
  my ( $timeout, $cmdRef, @args ) = @_;
  eval {
    local $SIG{ALRM} = sub { die "TIMED_OUT_CLIENT_SIDE" };
    alarm($timeout);
    $cmdRef->(@args);
    alarm(0);
  };
  return $@ if $@;
  return;
}

sub putWithTimeout {
  my ( $publisher, $timeout, @args ) = @_;

  my $start = time();
  my $err = executeWithTimeout( $timeout, sub { $publisher->put(@args) } );
  my $timeTaken = time() - $start;

  if ($err) {
    say STDERR "Failed to put message after $timeTaken seconds due to: $err";
    return $err;
  }

  if( $publisher->error ) {
    say STDERR "Failed to put message after $timeTaken seconds due to:" . $publisher->error;
    return $publisher->error;
  }

  say "Put message in $timeTaken seconds";

  return;
}

sub reserveWithTimeout {
  my ( $consumer, $timeout ) = @_;
  my $result;

  # We need to double the timeout because the reserve command will block until a job is available
  # And we only want to error if the beanstalkd server timeout isn't respected
  my $start = time();
  my $err =
    executeWithTimeout( $timeout * 2, sub { $result = $consumer->reserve($timeout) } );
  my $timeTaken = time() - $start;

  if ($err) {
    say STDERR "Failed to reserve job after $timeTaken seconds due to: $err";
    return ( $err, undef );
  }
  if( $consumer->error ) {
    say STDERR "Failed to reserve job after $timeTaken seconds due to: " . $consumer->error;
    return ( $consumer->error, undef );
  }

  say "Reserved job with id " . $result->id . " in $timeTaken seconds";
  return ( undef, $result );
}

sub deleteWithTimeout {
  my ( $job, $timeout ) = @_;

  my $start = time();
  my $err = executeWithTimeout( $timeout, sub { $job->delete() } );
  my $timeTaken = time() - $start;

  if($err) {
    say STDERR "Failed to delete job with id ". $job->id . " in $timeTaken seconds due to: $err";
    return $err;
  }

  if( $job->client->error ) {
    say STDERR "Failed to delete job  ". $job->id . " in $timeTaken seconds due to: " . $job->client->error;
    return $job->client->error;
  }

  say "Deleted job with id " . $job->id . " in $timeTaken seconds";

  return;
}

sub releaseWithTimeout {
  my ( $job, $timeout ) = @_;
  my $start = time();
  my $err = executeWithTimeout( $timeout, sub { $job->release() } );
  my $timeTaken = time() - $start;

  if($err) {
    say STDERR "Failed to release job after $timeTaken due to: $err";
    return $err;
  }

  if( $job->client->error ) {
    say STDERR "Failed to release job after $timeTaken due to: $err";
    return $job->client->error;
  }

  say "Released job with id " . $job->id . " in $timeTaken seconds";

  return;
}

sub statsWithTimeout {
  my ( $job, $timeout ) = @_;
  my $result;

  my $start = time();
  my $err = executeWithTimeout( $timeout, sub { $result = $job->stats() } );
  my $timeTaken = time() - $start;
  
  if($err) {
    say STDERR "Failed to get job stats after $timeTaken due to: $err";
    return ( $err, undef );
  }

  if( $job->client->error ) {
    say STDERR "Failed to get job stats after $timeTaken due to: " . $job->client->error;
    return ( $job->client->error, undef );
  }

  return ( undef, $result );
}

sub touchWithTimeout {
  my ( $job, $timeout ) = @_;
  my $err = executeWithTimeout( $timeout, sub { $job->touch() } );

  if($err) {
    say STDERR "Failed to touch job with error: $err";
    return $err;
  }

  if( $job->client->error ) {
    say STDERR "Failed to touch job with error: " . $job->client->error;
    return $job->client->error;
  }

  return;
}

sub handleJobFailure {
  my ( $job, $address, $err, $jobDataHref ) = @_;
  say STDERR "job " . $job->id . " failed due to $err";

  $err =~ s/\sat\s\w+\/\w+.*\sline\s\d+.*//;

  my $data = {
    event        => $FAILED,
    reason       => $err,
    queueId      => $job->id,
    submissionId => $jobDataHref->{submissionId}
  };

  my $beanstalkEvents =
    _getProducerClient( $conf->{beanstalkd}{addresses}[0], $queueConfig->{events} );

  my $error = putWithTimeout( $beanstalkEvents, $BEANSTALKD_RESERVE_TIMEOUT,
    { priority => 0, data => encode_json($data) } );

  if ($error) {
    say STDERR "Failed to send $FAILED message due to: $error";
  }

  my $jobShouldBeReleased = $error ? 1 : 0;
  
  my $connectError = connectJob( $job, $address );
  if($connectError) {
    say STDERR "Failed to connect to job with id " . $job->id . " with error $connectError";
  }

  # We will still try to release the job, even if we failed to connect to it, in case the connect error was transient
  # The error, if any will be logged

  releaseOrDeleteJob( $job, $jobShouldBeReleased, $BEANSTALKD_RESERVE_TIMEOUT );
}

sub handleJobCompletion {
  my ( $job, $address, $jobDataHref, $outputFileNamesHashRef, $totalAnnotated,
    $totalSkipped )
    = @_;
  my $data = {
    event        => $COMPLETED,
    queueId      => $job->id,
    submissionId => $jobDataHref->{submissionId},
    results      => {
      outputFileNames => $outputFileNamesHashRef,
      totalAnnotated  => $totalAnnotated,
      totalSkipped    => $totalSkipped
    }
  };

  say "Finished job with id " . $job->id . " with data:";
  p $data;

  my $beanstalkEvents =
    _getProducerClient( $conf->{beanstalkd}{addresses}[0], $queueConfig->{events} );

  my $error = putWithTimeout( $beanstalkEvents, $BEANSTALKD_RESERVE_TIMEOUT,
    { priority => 0, data => encode_json($data) } );

  if ($error) {
    say STDERR "Releasing job because we failed to put the completion message: $error";
  }

  my $jobShouldBeReleased = $error ? 1 : 0;
  my $connectError = connectJob( $job, $address );
  if($connectError) {
    say STDERR "Failed to connect to job with id " . $job->id . " with error $connectError";
  }

  # We will still try to release/delete the job, even if we failed to connect to it, in case the connect error was transient
  my $terminalError =
    releaseOrDeleteJob( $job, $jobShouldBeReleased, $BEANSTALKD_CONNECT_TIMEOUT );

  if (!$terminalError) {
    say "Job with id " . $job->id . " completed successfully";
  }
}

sub releaseOrDeleteJob {
  my ( $job, $shouldRelease, $timeout ) = @_;

  if ($shouldRelease) {
    return releaseWithTimeout( $job, $timeout );
  }

  return deleteWithTimeout( $job, $timeout );
}

sub runAnnotation {
  my ($jobDataHref) = @_;

  my $annotateInstance = Seq->new_with_config($jobDataHref);
  my ( $err, $outputFileNamesHashRef, $totalAnnotated, $totalSkipped ) =
    $annotateInstance->annotate();

  return {
    output    => $outputFileNamesHashRef,
    annotated => $totalAnnotated,
    skipped   => $totalSkipped,
    err       => $err
  };
}

# Main worker loop
my $lastAddressIndex = 0;
while (1) {
  # round robin between beanstalkd servers, taking $conf->{beanstalkd}{addresses} in turn
  # looping back to the first server after the last one
  my $address = $conf->{beanstalkd}{addresses}[$lastAddressIndex];
  $lastAddressIndex =
    ( $lastAddressIndex + 1 ) % scalar( @{ $conf->{beanstalkd}{addresses} } );

  my $beanstalk = _getConsumerClient( $address, $queueConfig->{submission} );

  my ( $reserveErr, $job ) =
    reserveWithTimeout( $beanstalk, $BEANSTALKD_RESERVE_TIMEOUT );

  # Logged in reserveWithTimeout
  if ($reserveErr) {
    next;
  }

  if ( !$job ) {
    say STDERR "No job found, despite no error being thrown. This should not happen.";
    next;
  }

  my $jobDataHref;
  my ( $outputFileNamesHashRef, $totalAnnotated, $totalSkipped );
  my $err;
  try {
    $jobDataHref = decode_json( $job->data );
    say "Reserved job with id " . $job->id . " which contains:";
    p $jobDataHref;
    
    my ( $coerceErr, $inputHref ) = coerceInputs( $jobDataHref, $job->id );
    if ($coerceErr) {
      die $coerceErr;
    }

    my $beanstalkEvents = _getProducerClient( $address, $queueConfig->{events} );
    my $putErr          = putWithTimeout(
      $beanstalkEvents,
      $BEANSTALKD_RESERVE_TIMEOUT,
      {
        priority => 0,
        data     => encode_json(
          {
            event        => $STARTED,
            submissionId => $jobDataHref->{submissionId},
            queueId      => $job->id
          }
        )
      }
    );

    # This error is less critical, because progress messages
    # will set the API server state to "started"
    if ($putErr) {
      say STDERR "Failed to send started message: $err";
    }

    my $result = MCE::Shared->hash();
    my $done   = MCE::Shared->scalar();

    my $heartbeatClient = MCE::Hobo->create(
      sub {
        my $lastTouchTime = time();
        my $isDone        = $done->get();

        # We cannot wrap this in a try/catch, maybe because we're already nested in a try/catch
        # TODO 2024-07-09 @akotlar: investigate what happens if this loop dies
        while ( !$isDone ) {
          if ( time() - $lastTouchTime >= $BEANSTALKD_JOB_HEARTBEAT ) {
            $lastTouchTime = time();
            my $touchErr = touchWithTimeout( $job, $BEANSTALKD_RESERVE_TIMEOUT);

            if ( $touchErr ) {
              say STDERR "Failed to touch job with id "
                . $job->id
                . " with error $touchErr";
            }
            else {
              say "Touched job   with id " . $job->id;

              my ($statsErr, $res) = statsWithTimeout($job, $BEANSTALKD_RESERVE_TIMEOUT);

              if ( $statsErr ) {
                say STDERR "Failed to get job stats with error $statsErr" ;
              } else {
                p $res;
              }
            }
          }

          sleep(1);

          $isDone = $done->get();
        }

        return;
      }
    );

    # We must run the annotation in the main thread, because internally it runs MCE, and
    # it turns out that MCE does not enjoy running from within itself (there may be workarounds that we do not understand)
    my $res = runAnnotation($inputHref);
    $done->set(1);

    $heartbeatClient->join();

    if ( !$res ) {
      die "Failed to get result from annotation job";
    }
    if ( $res->{err} ) {
      die $res->{err};
    }
    if ( !$res->{output} ) {
      die "Failed to get output from annotation job";
    }
    $outputFileNamesHashRef = $res->{output};
    $totalAnnotated         = $res->{annotated};
    $totalSkipped           = $res->{skipped};
  }
  catch {
    $err = $_;
  };

  if ($err) {
    handleJobFailure( $job, $address, $err, $jobDataHref );
  }
  else {
    handleJobCompletion( $job, $address, $jobDataHref, $outputFileNamesHashRef,
      $totalAnnotated, $totalSkipped );
  }

  sleep(1);
}

sub coerceInputs {
  my $jobDetailsHref = shift;
  my $queueId        = shift;

  my %args;
  my $err;

  my %jobSpecificArgs;
  for my $key ( keys %requiredForAll ) {
    if ( !defined $jobDetailsHref->{ $requiredForAll{$key} } ) {
      $err = "Missing required key: $requiredForAll{$key} in job message";
      return ( $err, undef );
    }

    $jobSpecificArgs{$key} = $jobDetailsHref->{ $requiredForAll{$key} };
  }

  for my $key ( keys %$requiredForType ) {
    if ( !defined $jobDetailsHref->{ $requiredForType->{$key} } ) {
      $err = "Missing required key: $key in job message";
      return ( $err, undef );
    }

    $jobSpecificArgs{$key} = $jobDetailsHref->{ $requiredForType->{$key} };
  }

  my $configFilePath = getConfigFilePath( $jobSpecificArgs{assembly} );

  if ( !$configFilePath ) {
    $err = "Assembly $jobSpecificArgs{assembly} doesn't have corresponding config file";
    return ( $err, undef );
  }

  my %commmonArgs = (
    config    => $configFilePath,
    publisher => {
      server      => $conf->{beanstalkd}{addresses}[0],
      queue       => $queueConfig->{events},
      messageBase => {
        event        => $PROGRESS,
        queueId      => $queueId,
        submissionId => $jobDetailsHref->{submissionId},
        data         => undef,
      }
    },
    compress       => 1,
    verbose        => $verbose,
    run_statistics => 1,
    archive        => 0
  );

  if ($maxThreads) {
    $commmonArgs{maxThreads} = $maxThreads;
  }

  my %combined = ( %commmonArgs, %jobSpecificArgs );

  return ( undef, \%combined );
}

sub getConfigFilePath {
  my $assembly = shift;

  if ( exists $configFilePathHref->{$assembly} ) {
    return $configFilePathHref->{$assembly};
  }
  else {
    my @maybePath = glob( $configPathBaseDir . $assembly . ".y*ml" );
    if ( scalar @maybePath ) {
      if ( scalar @maybePath > 1 ) {
        #should log
        say "\n\nMore than 1 config path found, choosing first";
      }

      return $maybePath[0];
    }

    die "\n\nNo config path found for the assembly $assembly. Exiting\n\n";
  }
}
