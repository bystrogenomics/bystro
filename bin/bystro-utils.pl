#!/usr/bin/env perl

use 5.10.0;
use strict;
use warnings;

use lib './lib';

use Getopt::Long;
use Path::Tiny qw/path/;
use Pod::Usage;
use YAML::XS qw/LoadFile/;
use String::Strip qw/StripLTSpace/;

use DDP;

use Utils::CaddToBed;
use Utils::Fetch;
use Utils::LiftOverCadd;
use Utils::SortCadd;
use Utils::RenameTrack;
use Utils::FilterCadd;
use Utils::RefGeneXdbnsfp;

use Seq::Build;

# TODO: refactor to automatically call util by string value
# i.e: --util filterCadd launches Utils::FilterCadd
my (
  $yaml_config, $names, $sortCadd, $filterCadd, $renameTrack, $utilName,
  $help,        $liftOverCadd, $liftOverPath, $liftOverChainPath,
  $debug,       $overwrite, $fetch, $caddToBed, $compress, $toBed,
  $renameTrackTo, $verbose, $dryRunInsertions, $maxThreads,
);

# usage
GetOptions(
  'c|config=s'   => \$yaml_config,
  'n|name=s'     => \$names,
  'h|help'       => \$help,
  'u|util=s'     => \$utilName,
  'd|debug=i'      => \$debug,
  'o|overwrite'  => \$overwrite,
  'v|verbose=i' => \$verbose,
  'r|dryRun' => \$dryRunInsertions,
  'm|maxThreads=i' => \$maxThreads,
);

if ($help || !$yaml_config) {
  Pod::Usage::pod2usage();
}

if(!$names) {
  my $config = LoadFile($yaml_config);
  my @tracks;
  for my $track (@{$config->{tracks}{tracks}}) {
    my $hasUtils = !!$track->{utils};

    if($hasUtils) {
      push @tracks, $track->{name};
    }
  }

  $names = join(",", @tracks);
}

if(!$names) {
  say STDERR "No tracks found with 'utils' property";
}

say "Running utils for : " . $names;

for my $wantedName (split ',', $names) {
  # modifies in place
  StripLTSpace($wantedName);

  my $config = LoadFile($yaml_config);
  my $utilConfigs;
  my $trackIdx = 0;

  my %options = (
    config       => $yaml_config,
    name         => $wantedName,
    debug        => $debug,
    overwrite    => $overwrite || 0,
    verbose      => $verbose,
    dryRun       => $dryRunInsertions,
  );

  if($maxThreads) {
    $options{maxThreads} = $maxThreads;
  }

  for my $track (@{$config->{tracks}{tracks}}) {
    if($track->{name} eq $wantedName) {
      $utilConfigs = $track->{utils};
      last;
    }

    $trackIdx++;
  }

  if (!$utilConfigs) {
    die "The $wantedName track must have 'utils' property";
  }

  for(my $utilIdx = 0; $utilIdx < @$utilConfigs; $utilIdx++) {
    if($utilName && $utilConfigs->[$utilIdx]{name} ne $utilName) {
      next;
    }

    # config may be mutated, by the last utility
    $config = LoadFile($yaml_config);
    my $utilConfig = $config->{tracks}{tracks}[$trackIdx]{utils}[$utilIdx];

    my $utilName = $utilConfig->{name};
    say $utilName;

    # Uppercase the first letter of the utility class name
    # aka user may specify "fetch" and we grab Utils::Fetch
    my $className = 'Utils::' . uc( substr($utilName, 0, 1) ) . substr($utilName, 1, length($utilName) - 1);
    my $args = $utilConfig->{args} || {};

    my %finalOpts = (%options, %$args, (utilIdx => $utilIdx, utilName => $utilName));

    my $instance = $className->new(\%finalOpts);
    $instance->go();
  }
}

__END__

=head1 NAME

run_utils - Runs items in lib/Utils

=head1 SYNOPSIS

run_utils
  --config <yaml>
  --name <track>
  [--debug]
  [--verbose]
  [--maxThreads]
  [--dryRun]
  [--overwrite]
  [--help]

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
