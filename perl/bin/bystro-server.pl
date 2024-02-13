
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

use DDP;

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

my $requiredForType = { input_file => 'inputFilePath' };

say "Running Annotation queue server";

my $configPathBaseDir  = "config/";
my $configFilePathHref = {};

my $queueConfig = $conf->{beanstalkd}{tubes}{$type};

if ( !$queueConfig ) {
  die "Queue config format not recognized. Options are "
    . ( join( ', ', @{ keys %{ $conf->{beanstalkd}{tubes} } } ) );
}

my $beanstalk = Beanstalk::Client->new(
  {
    server          => $conf->{beanstalkd}{addresses}[0],
    default_tube    => $queueConfig->{submission},
    connect_timeout => 1,
    encoder         => sub { encode_json( \@_ ) },
    decoder         => sub { @{ decode_json(shift) } },
  }
);

my $beanstalkEvents = Beanstalk::Client->new(
  {
    server          => $conf->{beanstalkd}{addresses}[0],
    default_tube    => $queueConfig->{events},
    connect_timeout => 1,
    encoder         => sub { encode_json( \@_ ) },
    decoder         => sub { @{ decode_json(shift) } },
  }
);

while ( my $job = $beanstalk->reserve ) {
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
      say STDERR "job " . $job->id . " starting with inputHref:";
      p $inputHref;
    }

    my $annotate_instance = Seq->new_with_config($inputHref);

    ( $err, $outputFileNamesHashRef ) = $annotate_instance->annotate();

    if ( defined $debug ) {
      p $err;
    }
  }
  catch {
    $err = $_;
  };

  if ( defined $debug ) {
    p $err;
  }

  if ($err) {
    $err =~ s/\sat\s\w+\/\w+.*\sline\s\d+.*//;

    say "job " . $job->id . " failed";

    my $data = {
      event        => $FAILED,
      reason       => $err,
      queueId      => $job->id,
      submissionId => $jobDataHref->{submissionId},
    };

    $beanstalkEvents->put( { priority => 0, data => encode_json($data) } );

    $job->delete();

    if ( $beanstalkEvents->error && defined $debug ) {
      say STDERR "Beanstalkd last error:";
      p $beanstalkEvents->error;
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
    say STDERR "putting completiong event";
    p $data;
  }

  # Signal completion before completion actually occurs via delete
  # To be conservative; since after delete message is lost
  $beanstalkEvents->put( { priority => 0, data => encode_json($data) } );

  $job->delete();

  if ( $beanstalkEvents->error && defined $debug ) {
    say "Beanstalkd last error:";
    p $beanstalkEvents->error;
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
