
#!/usr/bin/env perl
# Name:           snpfile_annotate_mongo_redis_queue.pl
# Description:
# Date Created:   Wed Dec 24
# By:             Alex Kotlar
# Requires: Snpfile::AnnotatorBase

use 5.10.0;
use Cpanel::JSON::XS;

use strict;
use warnings;

use Try::Tiny;

use Log::Any::Adapter;
use File::Basename;
use Getopt::Long;

use DDP output => *STDOUT;

use Beanstalk::Client;

use YAML::XS qw/LoadFile/;

use Seq;
use Path::Tiny qw/path/;

my ( $verbose, $queueConfigPath, $maxThreads, $debug );

GetOptions(
  'v|verbose=i'     => \$verbose,
  'd|debug'         => \$debug,
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

my $BEANSTALKD_TIMEOUT = 60;

sub _get_consumer_client {
  my $server_address = shift;
  my $tube           = shift;

  return Beanstalk::Client->new(
    {
      server          => $server_address,
      default_tube    => $tube,
      connect_timeout => 10,
      encoder         => sub { encode_json( \@_ ) },
      decoder         => sub { @{ decode_json(shift) } },
    }
  );
}

sub _get_producer_client {
  my $server_address = shift;
  my $tube           = shift;

  return Beanstalk::Client->new(
    {
      server          => $server_address,
      default_tube    => $tube,
      connect_timeout => 10,
      encoder         => sub { encode_json( \@_ ) },
      decoder         => sub { @{ decode_json(shift) } },
    }
  );
}

while (1) {
  my $beanstalk = _get_consumer_client( $conf->{beanstalkd}{addresses}[0],
    $queueConfig->{submission} );

  my $job = $beanstalk->reserve($BEANSTALKD_TIMEOUT);

  if ( !$job ) {
    next;
  }
  # Parallel ForkManager used only to throttle number of jobs run in parallel
  # cannot use run_on_finish with blocking reserves, use try catch instead
  # Also using forks helps clean up leaked memory from LMDB_File
  # Unfortunately, parallel fork manager doesn't play nicely with try tiny
  # prevents anything within the try from executing

  my $jobDataHref;
  my ( $err, $outputFileNamesHashRef );

  try {
    $jobDataHref = decode_json( $job->data );

    if ( defined $debug ) {
      say "Reserved job with id " . $job->id . " which contains:";
      p $jobDataHref;

      my $stats = $job->stats();
      say "stats are";
      p $stats;
    }

    ( $err, my $inputHref ) = coerceInputs( $jobDataHref, $job->id );

    if ($err) {
      die $err;
    }

    my $beanstalkEvents =
      _get_producer_client( $conf->{beanstalkd}{addresses}[0], $queueConfig->{events} );

    $beanstalkEvents->put(
      {
        priority => 0,
        data     => encode_json(
          {
            event        => $STARTED,
            submissionId => $jobDataHref->{submissionId},
            queueId      => $job->id,
          }
        )
      }
    );

    if ($debug) {
      say "job " . $job->id . " starting with inputHref:";
      p $inputHref;
    }

    my $annotate_instance = Seq->new_with_config($inputHref);

    ( $err, $outputFileNamesHashRef ) = $annotate_instance->annotate();

    if ($err) {
      die $err;
    }
  }
  catch {
    $err = $_;
  };

  if ($err) {
    say STDERR $err;

    $err =~ s/\sat\s\w+\/\w+.*\sline\s\d+.*//;

    say "job " . $job->id . " failed";

    my $data = {
      event        => $FAILED,
      reason       => $err,
      queueId      => $job->id,
      submissionId => $jobDataHref->{submissionId},
    };

    my $beanstalkEvents =
      _get_producer_client( $conf->{beanstalkd}{addresses}[0], $queueConfig->{events} );

    $beanstalkEvents->put( { priority => 0, data => encode_json($data) } );

    if ( $beanstalkEvents->error ) {
      say STDERR "Beanstalkd last error: " . $beanstalkEvents->error;
    }

    my $socket = $job->client->connect( $conf->{beanstalkd}{addresses}[0], 10 );

    if ( $job->client->error ) {
      say STDERR "Failed to connect to queue server with error " . $job->client->error;
    }
    elsif ( !$socket ) {
      say STDERR "Failed to connect to queue server for an unknown reason";
    }

    $job->delete();

    if ( $job->client->error ) {
      say STDERR "Failed to delete job with id "
        . $job->id
        . " with error "
        . $job->client->error;
    }

    next;
  }

  my $data = {
    event        => $COMPLETED,
    queueId      => $job->id,
    submissionId => $jobDataHref->{submissionId},
    results      => { outputFileNames => $outputFileNamesHashRef, }
  };

  if ( defined $debug ) {
    say "putting completion event";
    p $data;
  }

  my $beanstalkEvents =
    _get_producer_client( $conf->{beanstalkd}{addresses}[0], $queueConfig->{events} );

  # Signal completion before completion actually occurs via delete
  # To be conservative; since after delete message is lost
  $beanstalkEvents->put( { priority => 0, data => encode_json($data) } );

  if ( $beanstalkEvents->error ) {
    say STDERR "Beanstalkd last error: " . $beanstalkEvents->error;
  }

  my $socket = $job->client->connect( $conf->{beanstalkd}{addresses}[0], 10 );

  if ( $job->client->error ) {
    say STDERR "Failed to connect to queue server with error " . $job->client->error;
  }
  elsif ( !$socket ) {
    say STDERR "Failed to connect to queue server for an unknown reason";
  }

  $job->delete();

  if ( $job->client->error ) {
    say STDERR "Failed to delete job with id "
      . $job->id
      . " with error "
      . $job->client->error;
  }

  say "completed job with queue id " . $job->id;
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
