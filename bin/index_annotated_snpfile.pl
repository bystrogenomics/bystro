#!/usr/bin/env perl

use 5.10.0;
use Cpanel::JSON::XS;

use strict;
use warnings;

use lib './lib';

use SeqElastic;

use Getopt::Long;
use DDP;
use File::Basename;

use YAML::XS qw/LoadFile/;
use Path::Tiny qw/path/;
use Pod::Usage;

my (
  $indexName,   $verbose, $dryRunInsertions, $logDir, $debug, $annotatedFilePath,
  $typeName, $configPath, $err, $job, $connectionsPath,
);

# usage
GetOptions(
  'n|index_name=s'     => \$indexName,
  't|index_type=s'     => \$typeName,
  'v|verbose'    => \$verbose,
  'd|debug=i'      => \$debug,
  'a|annotated_file_path=s' => \$annotatedFilePath,
  'd|dry_run_insertions|dry|dryRun' => \$dryRunInsertions,
  'l|log_dir=s' => \$logDir,
  'indexConfig|config=s' => \$configPath,
  'c|connection=s' => \$connectionsPath,
);

unless ($indexName && $typeName && $annotatedFilePath) {
  Pod::Usage::pod2usage();
}

my ($sec,$min,$hour,$mday,$mon,$year,$wday,$yday,$isdst) = localtime();

$year += 1900;
#   # set log file
my $log_name = join '.', 'index', $indexName, "$mday\_$mon\_$year\_$hour\:$min\:$sec", 'log';

my $logPath = path($logDir || "/mnt/annotator_databases/logs/")->child($log_name)->absolute->stringify;
my $configPathBase = "config/";

my $connectionConfig = LoadFile($connectionsPath);


sub testConfig {
  my $testing = shift;

  my %indexConfig;
  $indexConfig{indexConfig} = LoadFile($configPathBase . 'hg19'. '.mapping.yml');
  $indexConfig{indexName} = "test" ;
  $indexConfig{indexType} = "638546fb31a5fd00209b915f" ;
  $indexConfig{inputDir} = "/seqant/user-data/6374cc633b670000204e3b19/638de6c491cde60020890444/output" ;

  %indexConfig = (%indexConfig, %$connectionConfig);

  return \%indexConfig;

}

my $input = testConfig();
  
#$input->{indexName} = $indexName ;
#$input->{indexType}  = $typeName ;
$input->{verbose} = $verbose ;
$input->{debug} = ($debug || 0) ;
$input->{annotatedFilePath} = $annotatedFilePath ;
$input->{logPath} = $logPath ;
#$input->{indexConfig} = $indexConfig->{indexConfig} ;
#$input->{connection} = $indexConfig->{connection} ;

if ($err) {
  if(defined $verbose || defined $debug) {
    say "job ". $job->id . " failed due to found error";
    p $err;}
}

my $app = SeqElastic->new($input);
$app->go;