#!/usr/bin/env perl

use 5.10.0;
use strict;
use warnings;

use Carp qw/ croak /;
use Getopt::Long;
use Path::Tiny qw/path/;
use Pod::Usage;
use Log::Any::Adapter;
use YAML::XS qw/ LoadFile /;

use DDP;

use Seq::Build;

my (
  $yaml_config,     $wantedType, $wantedName, $verbose,
  $maxThreads,      $help,       $wantedChr,  $dryRunInsertions,
  $logDir,          $debug,      $overwrite,  $delete,
  $regionTrackOnly, $skipCompletionCheck
);

$debug = 0;

# usage
GetOptions(
  'c|config=s'                    => \$yaml_config,
  't|type=s'                      => \$wantedType,
  'n|name=s'                      => \$wantedName,
  'v|verbose=i'                   => \$verbose,
  'h|help'                        => \$help,
  'd|debug=i'                     => \$debug,
  'o|overwrite'                   => \$overwrite,
  'chr=s'                         => \$wantedChr,
  'delete'                        => \$delete,
  'build_region_track_only'       => \$regionTrackOnly,
  'skip_completion_check'         => \$skipCompletionCheck,
  'dry_run_insertions|dry|dryRun' => \$dryRunInsertions,
  'log_dir=s'                     => \$logDir,
  'threads=i'                     => \$maxThreads
) or pod2usage(2);

if ($help) {
  pod2usage(1);
  exit;
}

unless ($yaml_config) {
  pod2usage("Error: --config is required");
}

if ($debug) {
  say STDERR "Running with the following parameters:";
  my $options = {
    config                  => $yaml_config,
    wantedChr               => $wantedChr,
    wantedType              => $wantedType,
    wantedName              => $wantedName,
    overwrite               => $overwrite || 0,
    debug                   => $debug     || 0,
    delete                  => !!$delete,
    build_region_track_only => !!$regionTrackOnly,
    skipCompletionCheck     => !!$skipCompletionCheck,
    dryRunInsertions        => !!$dryRunInsertions,
    logDir                  => $logDir,
    maxThreads              => $maxThreads,
    verbose                 => $verbose,
  };

  p $options;
}

# read config file to determine genome name for log and check validity
my $config_href = LoadFile($yaml_config);
# get absolute path for YAML file and db_location
$yaml_config = path($yaml_config)->absolute->stringify;

my ( $sec, $min, $hour, $mday, $mon, $year, $wday, $yday, $isdst ) = localtime();

$year += 1900;
#   # set log file
my $log_name =
     join '.', 'build', $config_href->{assembly}, $wantedType
  || $wantedName
  || 'allTracks', $wantedChr
  || 'allChr',
  "$mday\_$mon\_$year\_$hour\:$min\:$sec", 'log';

if ( !$logDir ) {
  $logDir = $config_href->{database_dir};

  # make or silently fail
  path($logDir)->mkpath();
}
my $logPath = path($logDir)->child($log_name)->absolute->stringify;

my $builder_options_href = {
  config                  => $yaml_config,
  wantedChr               => $wantedChr,
  wantedType              => $wantedType,
  wantedName              => $wantedName,
  overwrite               => $overwrite || 0,
  debug                   => $debug     || 0,
  logPath                 => $logPath,
  delete                  => !!$delete,
  build_region_track_only => !!$regionTrackOnly,
  skipCompletionCheck     => !!$skipCompletionCheck,
  dryRun                  => !!$dryRunInsertions,
  verbose                 => $verbose,
};

if ( defined $maxThreads ) {
  $builder_options_href->{maxThreads} = $maxThreads;
}

my $builder = Seq::Build->new_with_config($builder_options_href);

__END__

=head1 NAME

build_genome_assembly - Builds a binary genome assembly

=head1 SYNOPSIS

build_genome_assembly [options]

 Options:
   -c, --config                   YAML configuration file
   -t, --type                     Type of build (e.g., genome, conserv, transcript_db, gene_db, snp_db)
   -n, --name                     Name of the build
   -v, --verbose                  Verbosity level
   -h, --help                     Display this help message
   -d, --debug                    Debug level (default: 0)
   -o, --overwrite                For a given track, overwrite existing track values, rather than merging them
   --chr                          Chromosome to build (if applicable)
   --delete                       Delete the track instead of building it
   --build_region_track_only      Build region track only
   --skip_completion_check        Skip completion check
   --dry_run_insertions, --dry    Perform a dry run
   --log_dir                      Directory for log files
   --threads                      Number of threads to use

=head1 DESCRIPTION

C<build_genome_assembly.pl> takes a YAML configuration file and reads raw genomic
data that has been previously downloaded into the 'raw' folder to create the binary
index of the genome and associated annotations in the MongoDB instance.

=head1 OPTIONS

=over 8

=item B<-c>, B<--config>

config: A YAML genome assembly configuration file that specifies the various
tracks and data associated with the assembly. This is the same file that is
used by the Bystro Package to annotate VCF and SNP files.

=item B<-t>, B<--type>

type: Build all tracks in the configuration file with `type: <this_type>`.

=item B<-n>, B<--name>

name: Build the track specified in the configuration file with `name: <this_name>`.

=item B<--chr>

chr: Chromosome to build, if building gene or SNP; will build all if not specified.

=item B<-v>, B<--verbose>

verbose: Verbosity level (default: 0).

=item B<-d>, B<--debug>

debug: Debug level (default: 0).

=item B<-o>, B<--overwrite>

overwrite: For a given track, overwrite existing track values, rather than merging them

=item B<--delete>

delete: Delete the track instead of building it.

=item B<--build_region_track_only>

build_region_track_only: Build region track only.

=item B<--skip_completion_check>

skip_completion_check: Skip completion check.

=item B<--dry_run_insertions>, B<--dry>

dry_run_insertions: Perform a dry run

=item B<--log_dir>

log_dir: Directory for log files.

=item B<--threads>

threads: Number of threads to use.

=back

=head1 AUTHOR

Bystro Team

=head1 SEE ALSO

Bystro Package

=cut
