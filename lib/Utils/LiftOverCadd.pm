use 5.10.0;
use strict;
use warnings;

use lib '../';
# Takes a yaml file that defines one local file, and splits it on chromosome
# Only works for tab-delimitd files that have the c
package Utils::LiftOverCadd;

our $VERSION = '0.001';

use Mouse 2;
use namespace::autoclean;
use Types::Path::Tiny qw/AbsFile Path AbsDir/;
use Path::Tiny qw/path/;

use Seq::Tracks::Build::LocalFilesPaths;

use DDP;
use Parallel::ForkManager;

# _localFilesDir, _decodedConfig, compress, _wantedTrack, _setConfig, and logPath, 
extends 'Utils::Base';

########## Arguments accepted ##############
# Take the CADD file and make it a bed file
# the liftOver path is not AbsFile, so that 'liftOver' is valid (provided in $PATH)
has liftOverPath => (is => 'ro', isa => Path, coerce => 1, default => 'liftOver');
has liftOverChainPath => (is => 'ro', isa => AbsFile, coerce => 1, required => 1);

my $localFilesHandler = Seq::Tracks::Build::LocalFilesPaths->new();
sub liftOver {
  my $self = shift;

  my $liftOverExe = $self->liftOverPath;
  my $chainPath = $self->liftOverChainPath;

  $self->log('info', "Liftover path is $liftOverExe and chainPath is $chainPath");

  my $gzip = $self->gzip;

  my $localFilesPathsAref = $localFilesHandler->makeAbsolutePaths($self->_decodedConfig->{files_dir},
    $self->_wantedTrack->{name}, $self->_wantedTrack->{local_files});

  my $pm = Parallel::ForkManager->new(scalar @$localFilesPathsAref);

  if(!@$localFilesPathsAref) {
    $self->log('fatal', "No local files found");
  }

  my @finalOutPaths;

  for my $inPath (@$localFilesPathsAref) {
    $self->log('info', "Beginning to lift over $inPath");

    my (undef, $isCompressed, $inFh) = $self->get_read_fh($inPath);

    my $inPathPart = $isCompressed ? substr( $inPath, 0, rindex($inPath, ".") )
      : $inPath;

    my $compressOutput = $isCompressed || $self->compress;

    # It's a bit confusing how to compress stderr on the fly alongside stdout
    # So just compress it (always) as a 2nd step
    my $unmappedPath = $inPathPart . ".unmapped.txt";
    my $liftedPath = $inPathPart . ".mapped" . ($compressOutput ? '.gz' : '');

    if(-e $liftedPath && -e $unmappedPath && !$self->overwrite) {
      $self->log('info', "$liftedPath and $unmappedPath exist, and overwrite not set. Skipping.");
      close $inFh;

      # Push so that we can update our local_files after loop finishes
      push @finalOutPaths, $liftedPath;

      next;
    }

    $self->log('info', "Set mapped out path as: $liftedPath");
    $self->log('info', "Set unmapped out path as: $unmappedPath");

    ################## Write the headers to the output file (prepend) ########
    my $versionLine = <$inFh>;
    my $headerLine = <$inFh>;
    chomp $versionLine;
    chomp $headerLine;

    my $outFh = $self->get_write_fh($liftedPath);
    say $outFh $versionLine;
    say $outFh $headerLine;
    close $outFh;

    $self->log('info', "Wrote version line: $versionLine");
    $self->log('info', "Wrote header line: $headerLine");

    $pm->start($liftedPath) and next;
      ################ Liftover #######################
      # Decompresses
      my $command;
      if(!$isCompressed) {
        $command = "$liftOverExe <(cat $inPath | tail -n +3) $chainPath /dev/stdout $unmappedPath -bedPlus=3 ";
        if($compressOutput) {
          $command .= "| $gzip -c - >> $liftedPath";
        } else {
          $command .= "| cat - >> $liftedPath";
        }
      } else {
        $command = "$liftOverExe <($gzip -d -c $inPath | tail -n +3) $chainPath /dev/stdout $unmappedPath -bedPlus=3 | $gzip -c - >> $liftedPath; $gzip $unmappedPath";
      }

      $self->log('info', "Beginning to exec command: $command");

      #Can't open and stream, limited shell expressions supported, subprocess is not
      my $exitStatus = system(("bash", "-c", $command));

      if($exitStatus != 0) {
        $self->log('fatal', "liftOver command for $inPath failed with exit status: $exitStatus");
      } else {
        $self->log('info', "Successfully completed liftOver with with exit status: $exitStatus");
      }
    $pm->finish($exitStatus);
  }

  $pm->run_on_finish(sub{
    my ($pid, $exitCode, $liftedPath) = @_;
    if($exitCode != 0) {
      $self->log('fatal', "$liftedPath failed liftOver");
      return;
    }

    push @finalOutPaths, path($liftedPath)->basename;
  });

  $pm->wait_all_children;

  $self->_wantedTrack->{local_files} = \@finalOutPaths;

  $self->_wantedTrack->{liftOverCadd_date} = $self->_dateOfRun;

  $self->_backupAndWriteConfig();
}

__PACKAGE__->meta->make_immutable;
1;
