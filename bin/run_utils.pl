#!/usr/bin/env perl

use 5.10.0;
use strict;
use warnings;

use lib './lib';

use Getopt::Long;
use Path::Tiny qw/path/;
use Pod::Usage;

use Utils::CaddToBed;
use Utils::Fetch;
use Utils::LiftOverCadd;
use Utils::SortCadd;
use Utils::RenameTrack;

use Seq::Build;

my (
  $yaml_config, $wantedName, $sortCadd, $renameTrack,
  $help,        $liftOverCadd, $liftOverCadd_path, $liftOverChainPath,
  $debug,       $overwrite, $fetch, $caddToBed, $compress, $toBed,
  $renameTrackTo, $verbose, $dryRunInsertions,
);

# usage
GetOptions(
  'c|config=s'   => \$yaml_config,
  'n|name=s'     => \$wantedName,
  'h|help'       => \$help,
  'd|debug=i'      => \$debug,
  'o|overwrite'  => \$overwrite,
  'fetch' => \$fetch,
  'caddToBed' => \$caddToBed,
  'sortCadd'  => \$sortCadd,
  'renameTrack'  => \$renameTrack,
  'liftOverCadd' => \$liftOverCadd,
  'compress' => \$compress,
  'liftOverPath=s' => \$liftOverCadd_path,
  'liftOverChainPath=s' => \$liftOverChainPath,
  'renameTo=s' => \$renameTrackTo,
  'verbose=i' => \$verbose,
  'dryRun' => \$dryRunInsertions,
);

if ( (!$fetch && !$caddToBed && !$liftOverCadd && !$sortCadd && !$renameTrack) || $help) {
  Pod::Usage::pod2usage(1);
  exit;
}

unless ($yaml_config) {
  Pod::Usage::pod2usage();
}

my %options = (
  config       => $yaml_config,
  name         => $wantedName || undef,
  debug        => $debug,
  overwrite    => $overwrite || 0,
  overwrite    => $overwrite || 0,
  liftOverPath => $liftOverCadd_path || '',
  liftOverChainPath => $liftOverChainPath || '',
  renameTo => $renameTrackTo,
  verbose => $verbose,
  dryRun => $dryRunInsertions,
);

if($compress) {
  $options{compress} = $compress;
}

# If user wants to split their local files, needs to happen before we build
# So that the YAML config file has a chance to update
if($caddToBed) {
  my $caddToBedRunner = Utils::CaddToBed->new(\%options);
  $caddToBedRunner->go();
}

if($fetch) {
  my $fetcher = Utils::Fetch->new(\%options);
  $fetcher->fetch();
}

if($liftOverCadd) {
  my $liftOverCadd = Utils::LiftOverCadd->new(\%options);
  $liftOverCadd->liftOver();
}

if($sortCadd) {
  my $sortCadder = Utils::SortCadd->new(\%options);
  $sortCadder->sort();
}

if($renameTrack) {
  say "renaming";
  my $renamer = Utils::RenameTrack->new(\%options);
  $renamer->go();
}

#say "done: " . $wantedType || $wantedName . $wantedChr ? ' for $wantedChr' : '';


__END__

=head1 NAME

run_utils - Runs items in lib/Utils

=head1 SYNOPSIS

run_utils
  --config <file>
  --compress
  --name
  [--debug]

=head1 DESCRIPTION

C<run_utils.pl> Lets you run utility functions in lib/Utils

=head1 OPTIONS

=over 8

=item B<-t>, B<--compress>

Flag to compress output files

=item B<-c>, B<--config>

Config: A YAML genome assembly configuration file that specifies the various
tracks and data associated with the assembly. This is the same file that is
used by the Seq Package to annotate snpfiles.

=item B<-w>, B<--name>

name: The name of the track in the YAML config file

=back

=head1 AUTHOR

Alex Kotlar

=head1 SEE ALSO

Seq Package

=cut
