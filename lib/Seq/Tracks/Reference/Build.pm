use 5.10.0;
use strict;
use warnings;

package Seq::Tracks::Reference::Build;

our $VERSION = '0.001';

# ABSTRACT: Builds a plain text genome used for binary genome creation
# VERSION

use Mouse 2;
use namespace::autoclean;
extends 'Seq::Tracks::Build';

use Seq::Tracks::Reference::MapBases;

use Parallel::ForkManager;
use DDP;

my $baseMapper = Seq::Tracks::Reference::MapBases->new();

sub buildTrack {
  my $self = shift;

  my $headerRegex = qr/\A>([\w\d]+)/;
  my $dataRegex = qr/(\A[ATCGNatcgn]+)\z/xms;

  my $pm = Parallel::ForkManager->new($self->max_threads);
  for my $file ( $self->allLocalFiles ) {
    # Expects 1 chr per file for n+1 files, or all chr in 1 file
    # Single writer to reduce copy-on-write db inflation
    $self->log('info', $self->name . ": Beginning building from $file");

    $pm->start($file) and next;
      my $fh = $self->get_read_fh($file);

      my $wantedChr;

      my $chrPosition = $self->based;

      my $count = 0;
      # Record which chromosomes we've worked on
      my %visitedChrs;
      FH_LOOP: while ( my $line = $fh->getline() ) {
        #super chomp; also helps us avoid weird characters in the fasta data string
        $line =~ s/^\s+|\s+$//g; #trim both ends, but not what's in between
        
        #could do check here for cadd default format
        #for now, let's assume that we put the CADD file into a wigfix format
        if ( $line =~ m/$headerRegex/ ) { #we found a wig header
          my $chr = $1;

          if(!$chr) {
            $self->log('fatal', $self->name . ": Require chr in fasta file headers");
            die $self->name . ": Require chr in fasta file headers";
          }
          
          # Our first header, or we found a new chromosome
          if( ($wantedChr && $wantedChr ne $chr) || !$wantedChr) {
            # We switched chromosomes
            if($wantedChr) {
              $self->db->cleanUp($wantedChr);
              $count = 0;
            }

            $wantedChr = $self->chrIsWanted($chr) && $self->completionMeta->okToBuild($chr) ? $chr : undef;
          }

          # We expect either one chr per file, or a multi-fasta file
          if(!$wantedChr) {
            if($self->chrPerFile) {
              $self->log('info', $self->name . ": chrs in file $file not wanted or previously completed. Skipping");

              last FH_LOOP;
            }

            next FH_LOOP;
          } 

          $visitedChrs{$wantedChr} //= 1;

          # Restart chrPosition count at 0, since assemblies are zero-based ($self->based defaults to 0)
          # (or something else if the user based: allows non-reference fasta-formatted sources)
          $chrPosition = $self->based;

          #don't store the header line
          next;
        }

        # If !$wantedChr we're likely in a mult-fasta file; could warn, but that spoils multi-threaded reads
        if ( !$wantedChr ) {
          next;
        }

        if( $line =~ $dataRegex ) {
          # Store the uppercase bases; how UCSC does it, how people likely expect it
          for my $char ( split '', uc($1) ) {
            #Args:             $chr,       $trackIndex,   $pos,         $trackValue,                 $mergeFunc, $skipComit
            $self->db->dbPatch($wantedChr, $self->dbName, $chrPosition, $baseMapper->baseMap->{$char}, undef, $count < $self->commitEvery);
            $count = $count < $self->commitEvery ? $count + 1 : 0;

            #must come after, to not be 1 off; assumes fasta file is sorted ascending contiguous 
            $chrPosition++;
          }
        }
      }

      #Commit, sync everything, including completion status, and release mmap
      $self->db->cleanUp();

      # Record completion. Safe because detected errors throw, kill process
      foreach ( keys %visitedChrs ) {
        $self->completionMeta->recordCompletion($_);
        $self->log('info', $self->name . ": recorded $_ completed from $file");
      }

      #13 is sigpipe, occurs if closing pipe before cat/pigz finishes
      if(!close($fh) && $? != 13) {
        $self->log('fatal', $self->name . ": failed to close $file with $! $?");
        die $self->name . ": failed to close $file with $!";
      } else {
        $self->log('info', $self->name . ": closed $file with $?");
      }
    #exit with exit code 0; this only happens if successfully completed
    $pm->finish(0);
  }

  $pm->run_on_finish( sub {
    my ($pid, $exitCode, $fileName) = @_;
    if($exitCode != 0) {
      my $err = $self->name . "failed to build from $fileName: exit code $exitCode";
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
