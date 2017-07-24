#!/usr/bin/env perl

use 5.10.0;
use strict;
use warnings;
use lib './lib';
use SeqElastic;
use Getopt::Long;
use DDP;

use Path::Tiny qw/path/;
use Pod::Usage;

my (
  $indexName,   $verbose, $dryRunInsertions, $logDir, $debug, $annotatedFilePath,
  $typeName, $configPath,
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
  'config|config=s' => \$configPath,
);

unless ($indexName && $typeName && $annotatedFilePath) {
  Pod::Usage::pod2usage();
}

my ($sec,$min,$hour,$mday,$mon,$year,$wday,$yday,$isdst) = localtime();

$year += 1900;
#   # set log file
my $log_name = join '.', 'index', $indexName, "$mday\_$mon\_$year\_$hour\:$min\:$sec", 'log';

my $logPath = path($logDir || "/mnt/annotator_databases/logs/")->child($log_name)->absolute->stringify;

my $app = SeqElastic->new({
  indexName => $indexName,
  indexType  => $typeName,
  verbose => $verbose,
  debug => $debug || 0,
  annotatedFilePath => $annotatedFilePath,
  dryRun => $dryRunInsertions,
  logPath => $logPath,
  config => $configPath,
});

$app->go;

=head1 NAME

annotate_snpfile - annotates a snpfile using a given genome assembly specified
in a configuration file

=head1 SYNOPSIS

annotate_snpfile.pl --config <assembly config> --snp <snpfile> --out <file_ext> --type <snp_1, snp_2>

=head1 DESCRIPTION

C<annotate_snpfile.pl> takes a yaml configuration file and snpfile and gives
the annotations for the sites in the snpfile.

=head1 OPTIONS

=over 8

=item B<-s>, B<--snp>

Snp: snpfile

=item B<-r>, B<--type>

Type: version of snpfile: snp_1 or snp_2

=item B<-c>, B<--config>

Config: A YAML genome assembly configuration file that specifies the various
tracks and data associated with the assembly. This is the same file that is also
used by the Seq Package to build the binary genome without any alteration.

=item B<-o>, B<--out>

Output directory: This is the output director.

=item B<--overwrite>

Overwrite: Overwrite the annotation file if it exists.

=item B<--no_skip_chr>

No_Skip_Chr: Try to annotate all chromosomes in the snpfile and die if unable
to do so.


=back

=head1 AUTHOR

Alex Kotlar

=head1 SEE ALSO

Seq Package

=cut
