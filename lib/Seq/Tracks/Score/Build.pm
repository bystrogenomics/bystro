use 5.10.0;
use strict;
use warnings;

package Seq::Tracks::Score::Build;

our $VERSION = '0.001';

# ABSTRACT: Build a sparse track file
# VERSION

use Mouse 2;

use namespace::autoclean;
use Parallel::ForkManager;
use DDP;

extends 'Seq::Tracks::Build';

use Seq::Tracks::Score::Build::Round;

my $rounder = Seq::Tracks::Score::Build::Round->new();
# score track could potentially be 0 based
# http://www1.bioinf.uni-leipzig.de/UCSC/goldenPath/help/wiggle.html
# if it is the BED format version of the WIG format.
has '+based' => (
  default => 1,
);

sub buildTrack{
  my $self = shift;

  my $fStep = 'fixedStep';
  my $vStep = 'variableStep';
  my $headerRegex = qr/^($fStep|$vStep)\s+chrom=(\S+)\s+start=(\d+)\s+step=(\d+)/;
    
  my @allChrs = $self->allLocalFiles;
  
  #Can't just set to 0, because then the completion code in run_on_finish won't run
  my $pm = Parallel::ForkManager->new($self->max_threads);

  for my $file ( $self->allLocalFiles ) {
    $self->log('info', $self->name . ": beginning to build from $file");

    $pm->start($file) and next; 
      unless ( -f $file ) {
        $self->log('fatal', $self->name . ": $file doesn't exist");
        die $self->name . ": $file doesn't exist";
      }

      my $fh = $self->get_read_fh($file);

      my $wantedChr;
      my $chrPosition; # absolute by default, 0 index
      
      my $step;
      my $stepType;

      my $based = $self->based;

      # Which chromosomes we've seen, for recording completionMeta
      my %visitedChrs; 

      my $count = 0;
      FH_LOOP: while ( <$fh> ) {
        #super chomp; #trim both ends, but not what's in between
        $_ =~ s/^\s+|\s+$//g; 

        if ( $_ =~ m/$headerRegex/ ) {
          my $chr = $2;

          $step = $4;
          $stepType = $1;

          my $start = $3;
          
          if(!$chr && $step && $start && $stepType) {
            $self->log('fatal', $self->name . ": require chr, step, start, and step type fields in wig header");
            die $self->name . ": require chr, step, start, and step type fields in wig header";
          }

          if($stepType eq $vStep) {
            $self->log('fatal', $self->name . ": variable step not currently supported");
            die $self->name . ": variable step not currently supported";
          }

          if(!$wantedChr || ( $wantedChr && $wantedChr ne $chr) ) {
            if($wantedChr) {
              #Commit any remaining transactions, remove the db map from memory
              $self->db->cleanUp($wantedChr);
              $count = 0;
            }

            $wantedChr = $self->chrIsWanted($chr) && $self->completionMeta->okToBuild($chr) ? $chr : undef;
          }

          #use may give us one or many files
          if(!$wantedChr) {
            if($self->chrPerFile) {
              $self->log('info', $self->name . ": chrs in file $file not wanted or previously completed. Skipping");

              last FH_LOOP;
            }

            next FH_LOOP;
          } 

          # take the offset into account
          $chrPosition = $start - $based;

          # Record what we've seen
          $visitedChrs{$wantedChr} //= 1;

          #don't store the header in the database
          next;
        }

        # there could be more than one chr defined per file, just skip 
        # until we get to what we want
        if ( !$wantedChr ) {
          next;
        }

        #Args:             $chr,       $trackIndex,   $pos,         $trackValue,        $mergeFunc, $skipCommit
        $self->db->dbPatch($wantedChr, $self->dbName, $chrPosition, $rounder->round($_), undef, $count < $self->commitEvery);
        $count = $count < $self->commitEvery ? $count + 1 : 0;

        #this must come AFTER we store the position's data in db, since we have a starting pos
        $chrPosition += $step;
      }

      #Commit, sync everything, including completion status, and release mmap
      $self->db->cleanUp();

      # Record completion. Safe because detected errors throw, kill process
      foreach (keys %visitedChrs) {
        $self->completionMeta->recordCompletion($_);
        $self->log('info', $self->name . ": recorded $_ completed from $file");
      }

      if(!close($fh) && $? != 13) {
        $self->log('fatal', $self->name . ": failed to close $file with $! ($?)");
        die $self->name . ": failed to close $file with $!";
      } else {
        $self->log('info', $self->name . ": closed $file with $?");
      }
    $pm->finish(0);
  }

  $pm->run_on_finish(sub {
    my ($pid, $exitCode, $fileName) = @_;

    if($exitCode != 0) {
      my $err = $self->name . ": failed building from $fileName: exit code $exitCode";
      $self->log('fatal', $err);
      die $err;
    }

    $self->log('info', $self->name . ": completed building from $fileName");
  });
  
  $pm->wait_all_children;

  return;
};

__PACKAGE__->meta->make_immutable;

1;