
#!/usr/bin/env perl
# Name:           snpfile_annotate_mongo_redis_queue.pl
# Description:
# Date Created:   Wed Dec 24
# By:             Alex Kotlar
# Requires: Snpfile::AnnotatorBase

#Todo: Handle job expiration (what happens when job:id expired; make sure no other job operations happen, let Node know via sess:?)
#There may be much more performant ways of handling this without loss of reliability; loook at just storing entire message in perl, and relying on decode_json
#Todo: (Probably in Node.js): add failed jobs, and those stuck in processingJobs list for too long, back into job queue, for N attempts (stored in jobs:jobID)
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
# use AnyEvent;
# use AnyEvent::PocketIO::Client;
#use Sys::Info;
#use Sys::Info::Constants qw( :device_cpu )
#for choosing max connections based on available resources

# max of 1 job at a time for now

# usage
# Debug is like verbose, but applies only for the beanstalk_queue_server.pl program
# is not passed through to Seq.pm
my ($verbose, $type, $queueConfigPath, $connectionConfigPath, $maxThreads, $debug);

GetOptions(
  'v|verbose=i'   => \$verbose,
  'd|debug'    => \$debug,
  't|type=s'     => \$type,
  'q|queueConfig=s'   => \$queueConfigPath,
  'c|connectionConfig=s'   => \$connectionConfigPath,
  'maxThreads=i' => \$maxThreads,
);

my $conf = LoadFile($queueConfigPath);

# Beanstalk servers will be sharded
my $beanstalkHost  = $conf->{beanstalk_host_1};
my $beanstalkPort  = $conf->{beanstalk_port_1};

# for jobID specific pings
# my $annotationStatusChannelBase  = 'annotationStatus:';

# The properties that we accept from the worker caller
my %requiredForAll = (
  output_file_base => 'outputBasePath',
  assembly => 'assembly',
);

# Job dependent; one of these is required by the program this worker calls
my %requiredByType = (
  'saveFromQuery' => {
    inputQueryBody => 'queryBody',
    fieldNames => 'fieldNames',
    indexName => 'indexName',
    indexType => 'indexType',
  },
  'annotation' => {
    input_file => 'inputFilePath',
  },
  'export' => {
    to => 'to',
    sampleList => 'sampleList',
    assembly => 'assembly',
  }
);

my %optionalForType = (
  'saveFromQuery' => {
    indexConfig => 'indexConfig',
    pipeline => 'pipeline',
  }
);

say "Running queue server of type: $type";

my $configPathBaseDir = "config/";
my $configFilePathHref = {};

my $queueConfig = $conf->{beanstalkd}{tubes}{$type};

if(!$queueConfig) {
  die "$type not recognized. Options are " . ( join(', ', @{keys %{$conf->{beanstalkd}{tubes}}} ) );
}

my $beanstalk = Beanstalk::Client->new({
  server    => $conf->{beanstalkd}{host} . ':' . $conf->{beanstalkd}{port},
  default_tube => $queueConfig->{submission},
  connect_timeout => 1,
  encoder => sub { encode_json(\@_) },
  decoder => sub { @{decode_json(shift)} },
});

my $beanstalkEvents = Beanstalk::Client->new({
  server    => $conf->{beanstalkd}{host} . ':' . $conf->{beanstalkd}{port},
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

     # create the annotator
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

    my $annotate_instance;

    if($type eq 'annotation') {
      $annotate_instance = Seq->new_with_config($inputHref);
    } elsif($type eq 'saveFromQuery') {
      my $connectionConfig;

      if($connectionConfigPath) {
        $connectionConfig = LoadFile($connectionConfigPath);
      } else {
        die "Require connection config for saveFromQuery";
      }

      $inputHref = {%$inputHref, %$connectionConfig};

      p $inputHref;

      $annotate_instance = SeqFromQuery->new_with_config($inputHref);
    } elsif($type eq 'export') {
      $annotate_instance = Export->new_with_config($inputHref);
    }

    ($err, $outputFileNamesHashRef) = $annotate_instance->annotate();

  } catch {
    # Don't store the stack
    $err = $_; #substr($_, 0, index($_, 'at'));
  };

  if ($err) {
    say "job ". $job->id . " failed";

    if(defined $verbose || defined $debug) {
      p $err;
    }

    if(ref $err) {
      say STDERR "Elasticsearch error:";
      p $err;

      # TODO: Improve error handling, this doesn't work reliably
      if($err->{vars}{body}{status} && $err->{vars}{body}{status} == 400) {
        $err = "Query failed to parse";
      } else {
        $err = "Issue handling query";
      }
    }

    $beanstalkEvents->put( { priority => 0, data => encode_json({
      event => $events->{failed},
      reason => $err,
      queueID => $job->id,
      submissionID  => $jobDataHref->{submissionID},
    }) } );

    $job->bury();

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

#Here we may wish to read a json or yaml file containing argument mappings
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

  my $requiredForType = $requiredByType{$type};

  for my $key (keys %$requiredForType) {
    if(!defined $jobDetailsHref->{$requiredForType->{$key}}) {
      $err = "Missing required key: $key in job message";
      return ($err, undef);
    }

    $jobSpecificArgs{$key} = $jobDetailsHref->{$requiredForType->{$key}};
  }

  my $optionalForType = $optionalForType{$type};

  for my $key (keys %$optionalForType) {
    if(defined $jobDetailsHref->{$optionalForType->{$key}}) {
      $jobSpecificArgs{$key} = $jobDetailsHref->{$optionalForType->{$key}};
    }
  }

  my $configFilePath = getConfigFilePath($jobSpecificArgs{assembly});

  if(!$configFilePath) {
    $err = "Assembly $jobSpecificArgs{assembly} doesn't have corresponding config file";
    return ($err, undef);
  }

  my %commmonArgs = (
    config             => $configFilePath,
    publisher => {
      server => $conf->{beanstalkd}{host} . ':' . $conf->{beanstalkd}{port},
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
    $commmonArgs{maxThreads} = $maxThreads;
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
    #throws the error
    #should log here
  }
}
1;