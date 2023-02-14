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
use Capture::Tiny ':all';
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

use SeqElastic;

# usage
# Debug like verbose, but isn't passed through to SeqElastic
my ($verbose, $queueConfigPath, $connectionConfigPath, $maxThreads, $debug);

GetOptions(
  'v|verbose=i'   => \$verbose,
  'd|debug'      => \$debug,
  'q|queueConfig=s'   => \$queueConfigPath,
  'c|connectionConfig=s' => \$connectionConfigPath,
  'm|max_threads=i' => \$maxThreads,
);

my $conf = LoadFile($queueConfigPath);
my $connectionConfig = LoadFile($connectionConfigPath);

# Beanstalk servers will be sharded
my $beanstalkHost  = $conf->{beanstalk_host_1};
my $beanstalkPort  = $conf->{beanstalk_port_1};

# Required fields
# The annotation_file_path is constructed from inputDir, inputFileNames by SeqElastic
my @requiredJobFields = qw/indexName indexType inputDir inputFileNames assembly/;

my $configPathBaseDir = "config/";
my $configFilePathHref = {};



my $beanstalk = Beanstalk::Client->new({
  server    => $conf->{beanstalkd}{host} . ':' . $conf->{beanstalkd}{port},
  default_tube => $conf->{beanstalkd}{tubes}{index}{submission},
  connect_timeout => 1,
  encoder => sub { encode_json(\@_) },
  decoder => sub { @{decode_json(shift)} },
});

my $beanstalkEvents = Beanstalk::Client->new({
  server    => $conf->{beanstalkd}{host} . ':' . $conf->{beanstalkd}{port},
  default_tube => $conf->{beanstalkd}{tubes}{index}{events},
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
  my ($err, $fieldNames, $searchConfigHashRef);

  try {
    $jobDataHref = decode_json( $job->data );

    $beanstalkEvents->put({ priority => 0, data => encode_json{
      event => $events->{started},
      submissionID => $jobDataHref->{submissionID},
      queueID => $job->id,
    }  } );

    ($err, $fieldNames, $searchConfigHashRef) = handleJob($jobDataHref, $job->id);
  } catch {
    $err = $_;
  };

  if ($err) {
    if(defined $verbose || defined $debug) {
      say "job ". $job->id . " failed due to found error";
      p $err;
    }

    my $message;

    if(ref $err eq 'Search::Elasticsearch::Error::Request') {
      $message = $err->{vars}{body}{error}{reason};
    }

    $beanstalkEvents->put( { priority => 0, data => encode_json({
      event => $events->{failed},
      submissionID => $jobDataHref->{submissionID},
      reason => $message,
      queueID => $job->id,
    }) } );

    $job->bury; 

    next;
  }

  # Signal completion before completion actually occurs via delete
  # To be conservative; since after delete message is lost
  $beanstalkEvents->put({ priority => 0, data =>  encode_json({
    event => $events->{completed},
    submissionID => $jobDataHref->{submissionID},
    queueID => $job->id,
    fieldNames => $fieldNames,
    indexConfig => $searchConfigHashRef,
  }) } );

  $beanstalk->delete($job->id);

  say "fake completed job with queue id " . $job->id;
}

sub handleJob {
  my $submittedJob = shift;
  my $queueID = shift;

  my $failed;

  my ($err, $inputHref) = coerceInputs($submittedJob);

  if($err) {
    say STDERR $err;
    return ($err, undef);
  }

  my $log_name = join '.', 'index', 'indexName', $inputHref->{indexName}, 'log';
  my $logPath = File::Spec->rel2abs( ".", $log_name );

  if($maxThreads) {
    $inputHref->{max_threads} = $maxThreads;
  }

  $inputHref->{logPath} = $logPath;
  $inputHref->{verbose} = $verbose;
  $inputHref->{debug} = $debug;
  $inputHref->{publisher} = {
    server => $conf->{beanstalkd}{host} . ':' . $conf->{beanstalkd}{port},
    queue  => $conf->{beanstalkd}{tubes}{index}{events},
    messageBase => {
      event => $events->{progress},
      submissionID => $submittedJob->{submissionID},
      queueID => $queueID,
      data => undef,
    }
  };

  my $assemblyMap = ($configPathBaseDir . $submittedJob->{assembly} . '.mapping.yml');
  my $inputFileName = ($submittedJob->{inputDir} . "/"  . $submittedJob->{inputFileNames}->{archived});
  my $indexName = $inputHref->{indexName};

  my $cmd = "go run index/go/simple_parser.go -in \"$inputFileName\" -index \"$assemblyMap\" -connection \"$connectionConfigPath\" -name \"$indexName\" -http";

  if(defined $verbose || defined $debug) {
    say "\nin handle job, jobData is";
    p $submittedJob;
    say "writing beanstalk index queue log file here: $logPath\n";
    say "\ncmd is `$cmd`";
  }

  my $stdout;
  my $stderr;
  my $exit;

  ($stdout, $stderr, $exit) = capture {
    system($cmd);
  };

  if($exit != 0) {
    say "EXIT WAS $exit";
    return ($stderr, undef, undef);
  }

  my $output_json = decode_json($stdout);
  p $stderr;

  return (undef, $output_json->{'fieldNames'}, $output_json->{'indexConfig'});
}

#Here we may wish to read a json or yaml file containing argument mappings
sub coerceInputs {
  my $jobDetailsHref = shift;

  my %return;
  for my $fieldName (@requiredJobFields) {
    if(!defined $jobDetailsHref->{$fieldName}) {
      say STDERR "$fieldName required";
      return ("$fieldName required", undef);
    }

    $return{$fieldName} = $jobDetailsHref->{$fieldName};
  }

  $return{indexConfig} = LoadFile($configPathBaseDir . $jobDetailsHref->{assembly} . '.mapping.yml');
  %return = (%return, %$connectionConfig);

  return (undef, \%return);
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
