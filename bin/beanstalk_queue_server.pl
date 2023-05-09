
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

use lib './lib';

use Log::Any::Adapter;
use File::Basename;
use Getopt::Long;

use DDP;

use Beanstalk::Client;

use YAML::XS qw/LoadFile/;

use Seq;
use SeqFromQuery;
use Path::Tiny qw/path/;

my ($verbose, $queueConfigPath, $connectionConfigPath, $maxThreads, $debug);

GetOptions(
  'v|verbose=i'   => \$verbose,
  'd|debug'    => \$debug,
  'q|queueConfig=s'   => \$queueConfigPath,
  'c|connectionConfig=s'   => \$connectionConfigPath,
  'max_threads=i' => \$maxThreads,
);

my $type = 'annotation';

my $conf = LoadFile($queueConfigPath);

# The properties that we accept from the worker caller
my %requiredForAll = (
  output_file_base => 'outputBasePath',
  assembly => 'assembly',
);

my $requiredForType = {input_file => 'inputFilePath'};

say "Running Annotation queue server";

my $configPathBaseDir = "config/";
my $configFilePathHref = {};

my $queueConfig = $conf->{beanstalkd}{tubes}{$type};

if(!$queueConfig) {
  die "Queue config format not recognized. Options are " . ( join(', ', @{keys %{$conf->{beanstalkd}{tubes}}} ) );
}

my $beanstalk = Beanstalk::Client->new({
  server    => $conf->{beanstalkd}{address}[0],
  default_tube => $queueConfig->{submission},
  connect_timeout => 1,
  encoder => sub { encode_json(\@_) },
  decoder => sub { @{decode_json(shift)} },
});

my $beanstalkEvents = Beanstalk::Client->new({
  server    => $conf->{beanstalkd}{address}[0],
  default_tube => $queueConfig->{events},
  connect_timeout => 1,
  encoder => sub { encode_json(\@_) },
  decoder => sub { @{decode_json(shift)} },
});

my $events = $conf->{beanstalkd}{events};

while(my $job = $beanstalk->reserve) {
  # Parallel ForkManager used only to throttle number of jobs run in parallel
  # cannot use run_on_finish with blocking reserves, use try catch instead
  # Also using forks helps clean up leaked memory from LMDB_File
  # Unfortunately, parallel fork manager doesn't play nicely with try tiny
  # prevents anything within the try from executing

  my $jobDataHref;
  my ($err, $outputFileNamesHashRef);

  try {
    $jobDataHref = decode_json( $job->data );

    if(defined $verbose || defined $debug) {
      say "Reserved job with id " . $job->id . " which contains:";
      p $jobDataHref;

      my $stats = $job->stats();
      say "stats are";
      p $stats;
    }

    ($err, my $inputHref) = coerceInputs($jobDataHref, $job->id);

    if($err) {
      die $err;
    }

    my $configData = LoadFile($inputHref->{config});

    # Hide the server paths in the config we send back;
    # Those represent a security concern
    $configData->{files_dir} = 'hidden';

    if($configData->{temp_dir}) {
      $configData->{temp_dir} = 'hidden';
    }

    $configData->{database_dir} = 'hidden';

    my $trackConfig;
    if(ref $configData->{tracks} eq 'ARRAY') {
      $trackConfig = $configData->{tracks};
    } else {
      # New version
      $trackConfig = $configData->{tracks}{tracks};
    }

    for my $track (@$trackConfig) {
      # Strip local_files of their directory names, for security reasons
      $track->{local_files} = [map { !$_ ? "" : path($_)->basename } @{$track->{local_files}}]
    }

    $beanstalkEvents->put({ priority => 0, data => encode_json({
      event => $events->{started},
      jobConfig => $configData,
      submissionID   => $jobDataHref->{submissionID},
      queueID => $job->id,
    })  });

    my $annotate_instance = Seq->new_with_config($inputHref);

    ($err, $outputFileNamesHashRef) = $annotate_instance->annotate();
  } catch {
    # Don't store the stack
    $err = $_;
  };

  if ($err) {
    say "job ". $job->id . " failed";

    if(defined $verbose || defined $debug) {
      p $err;
    }

    if(ref $err) {
      say STDERR "Elasticsearch error:";
      p $err;
    }

    $err = "There was an issue, sorry!";

    my $data = {
      event => $events->{failed},
      reason => $err,
      queueID => $job->id,
      submissionID   => $jobDataHref->{submissionID},
    };

    $beanstalkEvents->put( { priority => 0, data => encode_json($data) } );

    $job->delete();

    if($beanstalkEvents->error) {
      say STDERR "Beanstalkd last error:";
      p $beanstalkEvents->error;
    }

    next;
  }

  my $data = {
    event => $events->{completed},
    queueID => $job->id,
    submissionID   => $jobDataHref->{submissionID},
    results => {
      outputFileNames => $outputFileNamesHashRef,
    }
  };

  if(defined $debug) {
    say STDERR "putting completiong event";
    p $data;
  }

  # Signal completion before completion actually occurs via delete
  # To be conservative; since after delete message is lost
  $beanstalkEvents->put({ priority => 0, data => encode_json($data)} );

  $job->delete();

  if($beanstalkEvents->error) {
    say "Beanstalkd last error:";
    p $beanstalkEvents->error;
  }

  say "completed job with queue id " . $job->id;
}

sub coerceInputs {
  my $jobDetailsHref = shift;
  my $queueId = shift;

  my %args;
  my $err;

  my %jobSpecificArgs;
  for my $key (keys %requiredForAll) {
    if(!defined $jobDetailsHref->{$requiredForAll{$key}}) {
      $err = "Missing required key: $key in job message";
      return ($err, undef);
    }

    $jobSpecificArgs{$key} = $jobDetailsHref->{$requiredForAll{$key}};
  }

  for my $key (keys %$requiredForType) {
    if(!defined $jobDetailsHref->{$requiredForType->{$key}}) {
      $err = "Missing required key: $key in job message";
      return ($err, undef);
    }

    $jobSpecificArgs{$key} = $jobDetailsHref->{$requiredForType->{$key}};
  }

  my $configFilePath = getConfigFilePath($jobSpecificArgs{assembly});

  if(!$configFilePath) {
    $err = "Assembly $jobSpecificArgs{assembly} doesn't have corresponding config file";
    return ($err, undef);
  }

  my %commmonArgs = (
    config             => $configFilePath,
    publisher => {
      server => $conf->{beanstalkd}{address}[0],
      queue  => $queueConfig->{events},
      messageBase => {
        event => $events->{progress},
        queueID => $queueId,
        submissionID => $jobDetailsHref->{submissionID},
        data => undef,
      }
    },
    compress => 1,
    archive => 1,
    verbose => $verbose,
    run_statistics => 1,
  );

  if($maxThreads) {
    $commmonArgs{max_threads} = $maxThreads;
  }

  my %combined = (%commmonArgs, %jobSpecificArgs);

  return (undef, \%combined);
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
1;