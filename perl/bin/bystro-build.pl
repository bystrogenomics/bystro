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
  $yaml_config, $wantedType,      $wantedName, $verbose,
  $maxThreads,  $help,            $wantedChr,  $dryRunInsertions,
  $logDir,      $metaOnly,        $debug,      $overwrite,
  $delete,      $regionTrackOnly, $skipCompletionCheck
);

$debug = 0;

# usage
GetOptions(
  'c|config=s'                                => \$yaml_config,
  't|type|wantedType=s'                       => \$wantedType,
  'n|name|wantedName=s'                       => \$wantedName,
  'v|verbose=i'                               => \$verbose,
  'h|help'                                    => \$help,
  'd|debug=i'                                 => \$debug,
  'o|overwrite'                               => \$overwrite,
  'chr|wantedChr=s'                           => \$wantedChr,
  'delete'                                    => \$delete,
  'build_region_track_only'                   => \$regionTrackOnly,
  'skipCompletionCheck|skip_completion_check' => \$skipCompletionCheck,
  'dry_run_insertions|dry|dryRun'             => \$dryRunInsertions,
  'log_dir=s'                                 => \$logDir,
  'threads=i'                                 => \$maxThreads,
  'meta_only'                                 => \$metaOnly,
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
    metaOnly                => !!$metaOnly,
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
  meta_only               => !!$metaOnly,
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
   -t, --type, --wantedType       Type of build (e.g., genome, conserv, transcript_db, gene_db, snp_db)
   -n, --name, --wantedName       Name of the build
   -v, --verbose                  Verbosity level
   -h, --help                     Display this help message
   -d, --debug                    Debug level (default: 0)
   -o, --overwrite                Overwrite existing files
   --chr, --wantedChr             Chromosome to build (if applicable)
   --delete                       Delete mode
   --build_region_track_only      Build region track only
   --skipCompletionCheck          Skip completion check
   --dry_run_insertions, --dry    Perform a dry run of insertions
   --log_dir                      Directory for log files
   --threads                      Number of threads to use
   --meta_only                    Meta only mode

=head1 DESCRIPTION

C<build_genome_assembly.pl> takes a YAML configuration file and reads raw genomic
data that has been previously downloaded into the 'raw' folder to create the binary
index of the genome and associated annotations in the MongoDB instance.

=head1 OPTIONS

=over 8

=item B<-c>, B<--config>

Config: A YAML genome assembly configuration file that specifies the various
tracks and data associated with the assembly. This is the same file that is
used by the Seq Package to annotate VCF and SNP files.

=item B<-t>, B<--type>, B<--wantedType>

Type: Build all tracks in the configuration file with `type: <this_type>``.

=item B<-n>, B<--name>, B<--wantedName>

Name: Build the track specified in the configuration file with `name: <this_name>``.

=item B<--chr>, B<--wantedChr>

Wanted_chr: Chromosome to build, if building gene or SNP; will build all if not specified.

=item B<-v>, B<--verbose>

Verbose: Verbosity level (default: 0).

=item B<-d>, B<--debug>

Debug: Debug level (default: 0).

=item B<-o>, B<--overwrite>

Overwrite: Overwrite existing files.

=item B<--delete>

Delete: Delete mode.

=item B<--build_region_track_only>

Build_region_track_only: Build region track only.

=item B<--skipCompletionCheck>

SkipCompletionCheck: Skip completion check.

=item B<--dry_run_insertions>, B<--dry>, B<--dryRun>

Dry_run_insertions: Perform a dry run of insertions.

=item B<--log_dir>

Log_dir: Directory for log files.

=item B<--threads>

Threads: Number of threads to use.

=item B<--meta_only>

Meta_only: Meta only mode.

=back

=head1 AUTHOR

Bystro Team

=head1 SEE ALSO

Seq Package

=cut
