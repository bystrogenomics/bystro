#!/usr/bin/env perl

use 5.10.0;
use strict;
use warnings;

use lib './lib';

use Carp qw/ croak /;
use Getopt::Long;
use Path::Tiny qw/path/;
use Pod::Usage;
use Log::Any::Adapter;
use YAML::XS qw/ LoadFile /;

use DDP;

use Seq::Build;

my (
  $yaml_config, $wantedType,        $wantedName,        $verbose, $maxThreads,
  $help,        $wantedChr,        $dryRunInsertions,   $logDir,  $metaOnly,
  $debug,       $overwrite,  $delete, $regionTrackOnly, $skipCompletionCheck
);

$debug = 0;
# usage
GetOptions(
  'c|config=s'   => \$yaml_config,
  't|type|wantedType=s'     => \$wantedType,
  'n|name|wantedName=s'     => \$wantedName,
  'v|verbose=i'    => \$verbose,
  'h|help'       => \$help,
  'd|debug=i'      => \$debug,
  'o|overwrite=i'  => \$overwrite,
  'chr|wantedChr=s' => \$wantedChr,
  'delete' => \$delete,
  'build_region_track_only' => \$regionTrackOnly,
  'skip_completion_check' => \$skipCompletionCheck,
  'dry_run_insertions|dry|dryRun' => \$dryRunInsertions,
  'log_dir=s' => \$logDir,
  'max_threads=i' => \$maxThreads,
  'meta_only' => \$metaOnly,
);

if ($help) {
  Pod::Usage::pod2usage(1);
  exit;
}

unless ($yaml_config) {
  Pod::Usage::pod2usage();
}


# read config file to determine genome name for log and check validity
my $config_href = LoadFile($yaml_config);

# get absolute path for YAML file and db_location
$yaml_config = path($yaml_config)->absolute->stringify;

my ($sec,$min,$hour,$mday,$mon,$year,$wday,$yday,$isdst) = localtime();

$year += 1900;
#   # set log file
my $log_name = join '.', 'build', $config_href->{assembly}, $wantedType ||
$wantedName || 'allTracks', $wantedChr || 'allChr',
"$mday\_$mon\_$year\_$hour\:$min\:$sec", 'log';

if(!$logDir) {
  $logDir = $config_href->{database_dir};

  # make or silently fail
  path($logDir)->mkpath();
}
my $logPath = path($logDir)->child($log_name)->absolute->stringify;

my $builder_options_href = {
  config   => $yaml_config,
  wantedChr    => $wantedChr,
  wantedType   => $wantedType,
  wantedName   => $wantedName,
  overwrite    => $overwrite || 0,
  debug        => $debug || 0,
  logPath      => $logPath,
  delete       => !!$delete,
  build_region_track_only => !!$regionTrackOnly,
  skip_completion_check => !!$skipCompletionCheck,
  dryRun => !!$dryRunInsertions,
  meta_only => !!$metaOnly,
  verbose => $verbose,
};

if(defined $maxThreads) {
  $builder_options_href->{max_threads} = $maxThreads;
}
# my $log_file = path(".")->child($log_name)->absolute->stringify;
# Log::Any::Adapter->set( 'File', $log_file );

my $builder = Seq::Build->new_with_config($builder_options_href);

#say "done: " . $wantedType || $wantedName . $wantedChr ? ' for $wantedChr' : '';


__END__

=head1 NAME

build_genome_assembly - builds a binary genome assembly

=head1 SYNOPSIS

build_genome_assembly
  --config <file>
  --type <'genome', 'conserv', 'transcript_db', 'snp_db', 'gene_db'>
  [ --wanted_chr ]

=head1 DESCRIPTION

C<build_genome_assembly.pl> takes a yaml configuration file and reads raw genomic
data that has been previously downloaded into the 'raw' folder to create the binary
index of the genome and assocated annotations in the mongodb instance.

=head1 OPTIONS

=over 8

=item B<-t>, B<--type>

Type: A general command to start building; genome, conserv, transcript_db, gene_db
or snp_db.

=item B<-c>, B<--config>

Config: A YAML genome assembly configuration file that specifies the various
tracks and data associated with the assembly. This is the same file that is
used by the Seq Package to annotate snpfiles.

=item B<-w>, B<--wanted_chr>

Wanted_chr: chromosome to build, if building gene or snp; will build all if not
specified.

=back

=head1 AUTHOR

Thomas Wingo

=head1 SEE ALSO

Seq Package

=cut
