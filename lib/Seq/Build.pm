use 5.10.0;
use strict;
use warnings;

package Seq::Build;
use lib './lib';
our $VERSION = '0.001';

use DDP;
# ABSTRACT: A class for building all files associated with a genome assembly
# VERSION

=head1 DESCRIPTION

  @class Seq::Build
  Iterates all of the builders present in the config file
  And executes their buildTrack method
  Also guarantees that the reference track will be built first

  @example

=cut

use Mouse 2;
use namespace::autoclean;
extends 'Seq::Base';

use Seq::Tracks;
use Seq::Tracks::Base::Types;
use Utils::Base;
use List::Util qw/first/;
use YAML::XS qw/LoadFile Dump/;
use Path::Tiny qw/path/;
use Time::localtime;

has wantedType => (is => 'ro', isa => 'Maybe[Str]', lazy => 1, default => undef);

#TODO: allow building just one track, identified by name
has wantedName => (is => 'ro', isa => 'Maybe[Str]', lazy => 1, default => undef);

has meta_only => (is => 'ro', default => 0);

# The config file path, used to update the config file with build version date & author
has config => (is => 'ro', required => 1);

#Figures out what track type was asked for 
#and then builds that track by calling the tracks 
#"buildTrack" method
sub BUILD {
  my $self = shift;

  # From Seq::Base;
  my $tracks = $self->tracksObj;

  my $buildDate = Utils::Base::getDate();

  #http://stackoverflow.com/questions/1378221/how-can-i-get-name-of-the-user-executing-my-perl-script
  my $buildAuthor = $ENV{LOGNAME} || $ENV{USER} || getpwuid($<);
  # Meta tracks are built during instantiation, so if we only want to build the
  # meta data, we can return here safely.
  if($self->meta_only) {
    return;
  }
  
  my @builders;
  my @allBuilders = $tracks->allTrackBuilders;

  if($self->wantedType) {
    my @types = split(/,/, $self->wantedType);
    
    for my $type (@types) {
      my $buildersOfType = $tracks->getTrackBuildersByType($type);

      if(!defined $buildersOfType) {
        $self->log('fatal', "Track type \"$type\" not recognized");
        return;
      }
      
      push @builders, @$buildersOfType;
    }
  } elsif($self->wantedName) {
    my @names = split(/,/, $self->wantedName);
    
    for my $name (@names) {
      my $builderOfName = $tracks->getTrackBuilderByName($name);

      if(!defined $builderOfName) {
        $self->log('fatal', "Track name \"$name\" not recognized");
        return;
      }

      push @builders, $builderOfName;
    }
  } else {
    @builders = @allBuilders;

    #If we're building all tracks, reference should be first
    if($builders[0]->name ne $tracks->getRefTrackBuilder()->name) {
      $self->log('fatal', "Reference track should be listed first");
    }
  }

  #TODO: return error codes from the rest of the buildTrack methods
  my $decodedConfig = LoadFile($self->config);

  for my $builder (@builders) {
    $self->log('info', "Started building " . $builder->name . "\n");
    
    #TODO: implement errors for all tracks
    #Currently we expect buildTrack to die if it didn't properly complete
    $builder->buildTrack();

    my $track = first{$_->{name} eq $builder->name} @{$decodedConfig->{tracks}};

    $track->{build_date} = $buildDate;
    $track->{build_author} = $buildAuthor;
    $track->{version} = $track->{version} ? ++$track->{version} : 1;
    
    $self->log('info', "Finished building " . $builder->name . "\n");
  }

  $self->log('info', "finished building all requested tracks: " 
    . join(", ", map{ $_->name } @builders) . "\n");

  $decodedConfig->{build_date} = $buildDate;
  $decodedConfig->{build_author} = $buildAuthor;
  $decodedConfig->{version} = $decodedConfig->{version} ? ++$decodedConfig->{version} : 1;

  # If this is already a symlink, remove it
  if(-l $self->config) {
    unlink $self->config;
  } else {
    my $backupPath = $self->config . ".build-bak.$buildDate";
    if( system ("rm -f $backupPath; mv " . $self->config . " " . $self->config . ".build-bak.$buildDate" ) != 0 ) {
      $self->log('fatal', "Failed to back up " . $self->config);
    }
  }

  my $newConfigPath = $self->config . ".build.$buildDate";
  my $newConfigPathBase = path($newConfigPath)->basename;

  # Write a copy of the new config file to the database-containing folder
  $newConfigPath = path($decodedConfig->{database_dir})->child($newConfigPathBase)->stringify;
  open(my $fh, '>', $newConfigPath) or $self->log('fatal', "Couldn't open $newConfigPath for writing" );

  say $fh Dump($decodedConfig);

  close($fh);

  # Write a 2nd copy to the original path of the config file.
  open($fh, '>', $self->config);

  say $fh Dump($decodedConfig);

  close($fh);

  # Create a clean copy, free of file paths, for github
  $decodedConfig->{database_dir} = '~';
  $decodedConfig->{files} = '~';
  $decodedConfig->{temp_dir} = '~';

  $newConfigPathBase = path($self->config)->basename;
  $newConfigPathBase = substr($newConfigPathBase, 0, rindex($newConfigPathBase, '.')) . ".clean.yml";

  $newConfigPath = path($self->config)->parent->child($newConfigPathBase)->stringify;

  open($fh, '>', $newConfigPath);

  say $fh Dump($decodedConfig);
}

__PACKAGE__->meta->make_immutable;

1;
