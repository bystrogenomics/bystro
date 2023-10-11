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

# score track could potentially be 0 based
# http://www1.bioinf.uni-leipzig.de/UCSC/goldenPath/help/wiggle.html
# if it is the BED format version of the WIG format.
has '+based' => ( default => 1, );

has scalingFactor => ( is => 'ro', isa => 'Int', default => 100 );

sub BUILD {
  my $self = shift;

  $self->{_rounder} =
    Seq::Tracks::Score::Build::Round->new( { scalingFactor => $self->scalingFactor } );
}

sub buildTrack {
  my $self = shift;

  my $fStep       = 'fixedStep';
  my $vStep       = 'variableStep';
  my $headerRegex = qr/^($fStep|$vStep)\s+chrom=(\S+)\s+start=(\d+)\s+step=(\d+)/;

  my @allChrs = $self->allLocalFiles;

  #Can't just set to 0, because then the completion code in run_on_finish won't run
  my $pm = Parallel::ForkManager->new( $self->maxThreads );

  my %completedChrs;
  $pm->run_on_finish(
    sub {
      my ( $pid, $exitCode, $fileName, $exitSignal, $coreDump, $errOrChrs ) = @_;

      if ( $exitCode != 0 ) {
        my $err = $self->name
          . ": got exitCode $exitCode for $fileName: $exitSignal . Dump: $coreDump";

        $self->log( 'fatal', $err );
      }

      if ( $errOrChrs && ref $errOrChrs eq 'HASH' ) {
        for my $chr ( keys %$errOrChrs ) {
          if ( !$completedChrs{$chr} ) {
            $completedChrs{$chr} = [$fileName];
          }
          else {
            push @{ $completedChrs{$chr} }, $fileName;
          }
        }
      }

      #Only message that is different, in that we don't pass the $fileName
      $self->log( 'info', $self->name . ": completed building from $fileName" );
    }
  );

  for my $file ( $self->allLocalFiles ) {
    $self->log( 'info', $self->name . ": beginning to build from $file" );

    # Although this should be unnecessary, environments must be created
    # within the process that uses them
    # This provides a measure of safety
    $self->db->cleanUp();

    $pm->start($file) and next;
    my ( $err, undef, $fh ) = $self->getReadFh($file);

    if ($err) {
      $self->log( 'fatal', $self->name . ": $err" );
    }

    my $wantedChr;
    my $chrPosition; # absolute by default, 0 index

    my $step;
    my $stepType;

    my $based = $self->based;

    # Which chromosomes we've seen, for recording completionMeta
    my %visitedChrs;

    # We use "unsafe" writers, whose active cursors we need to track
    my $cursor;
    my $count = 0;

    FH_LOOP: while (<$fh>) {
      #super chomp; #trim both ends, but not what's in between
      $_ =~ s/^\s+|\s+$//g;

      if ( $_ =~ m/$headerRegex/ ) {
        my $chr = $2;

        $step     = $4;
        $stepType = $1;

        my $start = $3;

        if ( !$chr && $step && $start && $stepType ) {
          $self->log( 'fatal',
            $self->name . ": require chr, step, start, and step type fields in wig header" );
          die $self->name . ": require chr, step, start, and step type fields in wig header";
        }

        if ( $stepType eq $vStep ) {
          $self->log( 'fatal', $self->name . ": variable step not currently supported" );
          die $self->name . ": variable step not currently supported";
        }

        # Transforms $chr if it's not prepended with a 'chr' or is 'chrMT' or 'MT'
        # and checks against our list of wanted chromosomes
        $chr = $self->normalizedWantedChr->{$chr};

        # falsy value is ''
        if ( !defined $wantedChr || ( !defined $chr || $wantedChr ne $chr ) ) {
          if ( defined $wantedChr ) {
            #Commit any remaining transactions, remove the db map from memory
            #this also has the effect of closing all cursors
            $self->db->cleanUp();
            undef $cursor;

            $count = 0;
          }

          $wantedChr = $self->chrWantedAndIncomplete($chr);
        }

        # TODO: handle chrPerFile
        if ( !defined $wantedChr ) {
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
      if ( !defined $wantedChr ) {
        next;
      }

      $cursor //= $self->db->dbStartCursorTxn($wantedChr);

      #Args:                         $cursor,             $chr,       $trackIndex,   $pos,         $trackValue
      $self->db->dbPatchCursorUnsafe( $cursor, $wantedChr, $self->dbName, $chrPosition,
        $self->{_rounder}->round($_) );

      if ( $count > $self->commitEvery ) {
        $self->db->dbEndCursorTxn($wantedChr);
        undef $cursor;

        $count = 0;
      }

      $count++;

      #this must come AFTER we store the position's data in db, since we have a starting pos
      $chrPosition += $step;
    }

    #Commit, sync everything, including completion status, and release mmap
    $self->db->cleanUp();
    undef $cursor;

    $self->safeCloseBuilderFh( $fh, $file, 'fatal' );

    $pm->finish( 0, \%visitedChrs );
  }

  $pm->wait_all_children();

  for my $chr ( keys %completedChrs ) {
    $self->completionMeta->recordCompletion($chr);

    $self->log( 'info',
          $self->name
        . ": recorded $chr completed, from "
        . ( join( ",", @{ $completedChrs{$chr} } ) ) );
  }

  #TODO: figure out why this is necessary, even with DEMOLISH
  $self->db->cleanUp();

  return;
}

__PACKAGE__->meta->make_immutable;

1;
