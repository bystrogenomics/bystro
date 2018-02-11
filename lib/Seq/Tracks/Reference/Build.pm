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

  my %completedChrs;
  $pm->run_on_finish( sub {
    my ($pid, $exitCode, $fileName, $exitSignal, $coreDump, $errOrChrs) = @_;

    if($exitCode != 0) {
      my $err = $self->name . ": got exitCode $exitCode for $fileName: $exitSignal . Dump: $coreDump";

      $self->log('fatal', $err);
    }

    if($errOrChrs && ref $errOrChrs eq 'HASH') {
      for my $chr (keys %$errOrChrs) {
        if(!$completedChrs{$chr}) {
          $completedChrs{$chr} = [$fileName];
        } else {
          push @{$completedChrs{$chr}}, $fileName;
        }
      }
    }

    #Only message that is different, in that we don't pass the $fileName
    $self->log('info', $self->name . ": completed building from $fileName");
  });

  for my $file ($self->allLocalFiles) {
    # Expects 1 chr per file for n+1 files, or all chr in 1 file
    # Single writer to reduce copy-on-write db inflation
    $self->log('info', $self->name . ": Beginning building from $file");

    # Although this should be unnecessary, environments must be created
    # within the process that uses them
    # This provides a measure of safety
    $self->db->cleanUp();

    $pm->start($file) and next;
      my $fh = $self->get_read_fh($file);

      my $wantedChr;

      my $chrPosition = $self->based;

      my $count = 0;
      # Record which chromosomes we've worked on
      my %visitedChrs;

      my $cursor;

      FH_LOOP: while (my $line = $fh->getline()) {
        #super chomp; also helps us avoid weird characters in the fasta data string
        $line =~ s/^\s+|\s+$//g; #trim both ends, but not what's in between

        #could do check here for cadd default format
        #for now, let's assume that we put the CADD file into a wigfix format
        if ($line =~ m/$headerRegex/) { #we found a wig header
          my $chr = $1;

          if(!$chr) {
            $self->log('fatal', $self->name . ": Require chr in fasta file headers");
            die $self->name . ": Require chr in fasta file headers";
          }

          # Our first header, or we found a new chromosome
          if(!defined $wantedChr || $wantedChr ne $chr) {
            # We switched chromosomes
            if(defined $wantedChr) {
              # cleans up entire environment, commits/closes all cursors, syncs
              $self->db->cleanUp();
              undef $cursor;

              $count = 0;
            }

            $wantedChr = $self->chrWantedAndIncomplete($chr);
          }

          # We expect either one chr per file, or a multi-fasta file that is sorted and contiguous
          # TODO: Handle chrPerFile
          if(!defined $wantedChr) {
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
        if (!defined $wantedChr) {
          next;
        }

        if($line =~ $dataRegex) {
          # Store the uppercase bases; how UCSC does it, how people likely expect it
          for my $char (split '', uc($1)) {
            $cursor //= $self->db->dbStartCursorTxn($wantedChr);

            #Args:                         $cursor,             $chr,        $trackIndex,   $pos,         $newValue
            $self->db->dbPatchCursorUnsafe($cursor, $wantedChr, $self->dbName, $chrPosition, $baseMapper->baseMap->{$char});

            if($count > $self->commitEvery) {
              $self->db->dbEndCursorTxn($wantedChr);
              undef $cursor;

              $count = 0;
            }

            $count++;

            #must come after, to not be 1 off; assumes fasta file is sorted ascending contiguous
            $chrPosition++;
          }
        }
      }

      #Commit, sync everything, including completion status, commit cursors, and release mmap
      $self->db->cleanUp();
      undef $cursor;

      #13 is sigpipe, occurs if closing pipe before cat/pigz finishes
      if(!close($fh) && $? != 13) {
        $self->log('fatal', $self->name . ": failed to close $file with $! $?");
        die $self->name . ": failed to close $file with $!";
      } else {
        $self->log('info', $self->name . ": closed $file with $?");
      }

    #exit with exit code 0; this only happens if successfully completed
    $pm->finish(0, \%visitedChrs);
  }

  $pm->wait_all_children();

  for my $chr (keys %completedChrs) {
    $self->completionMeta->recordCompletion($chr);

    $self->log('info', $self->name . ": recorded $chr completed, from " . (join(",", @{$completedChrs{$chr}})));
  }

  #TODO: figure out why this is necessary, even with DEMOLISH
  $self->db->cleanUp();

  return;
};

__PACKAGE__->meta->make_immutable;

1;
