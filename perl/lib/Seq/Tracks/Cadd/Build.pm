use 5.10.0;
use strict;
use warnings;
# Adds cadd data to our main database
# Reads CADD's bed-like format
package Seq::Tracks::Cadd::Build;

use Mouse 2;
extends 'Seq::Tracks::Build';

use Seq::Tracks::Cadd::Order;
use Seq::Tracks::Score::Build::Round;
use Seq::Tracks;

use Scalar::Util qw/looks_like_number/;

# Cadd tracks seem to be 1 based (not well documented)
has '+based' => ( default => 1 );

# CADD files may not be sorted,
has sorted => ( is => 'ro', isa => 'Bool', lazy => 1, default => 0 );

has scalingFactor => ( is => 'ro', isa => 'Int', default => 10 );

my $order = Seq::Tracks::Cadd::Order->new();
$order = $order->order;

my $refTrack;

sub BUILD {
  my $self = shift;

  $self->{_rounder} =
    Seq::Tracks::Score::Build::Round->new( { scalingFactor => $self->scalingFactor } );
}
############## Version that does not assume positions in order ################
############## Will optimize for cases when sorted_guranteed truthy ###########
# TODO: refactor so that one function handles both main build, and the tail end
sub buildTrack {
  my $self = shift;

  # TODO: Remove side effects, or think about another initialization method
  # Unfortunately, it is better to call track getters here
  # Because builders may have side effects, like updating
  # the meta database
  # So we want to call builders BUILD methods first
  my $tracks = Seq::Tracks->new();
  $refTrack = $tracks->getRefTrackGetter();

  my $pm = Parallel::ForkManager->new( $self->maxThreads );

  ######Record completion status only if the process completed unimpeded ########
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

  #Perl is dramatically faster when splitting on a constant, so we assume '\t'
  if ( $self->delimiter ne '\t' && $self->delimiter ne "\t" ) {
    $self->log( "fatal", $self->name . ": requires delimiter to be \\t" );
  }

  my $missingValue = undef;

  # If we cannot rely on the cadd sorting order, we must use a defined
  # value for those bases that we skip, because we'll ned mergeFunc
  # to know when data was found for a position, and when it is truly missing
  # Because when CADD scores are not sorted, each chromosome-containing file
  # can potentially have any other chromosome's scores, meaning we may get
  # 6-mers or greater for a single position; when that happens the only
  # sensible solution is to store a missing value; undef would be nice,
  # but that will never get triggered, unless our database is configured to store
  # hashes instead of arrays; since a sparse array will contain undef/nil
  # for any track at that position that has not yet been inserted into the db
  # For now we require sorting to be guaranteed to simplify this code
  if ( !$self->sorted ) {
    $self->log( "fatal", $self->name . ": requires sorted to be true" );
  }

  for my $file ( @{ $self->local_files } ) {
    $self->log( 'info', $self->name . ": beginning building from $file" );

    # Although this should be unnecessary, environments must be created
    # within the process that uses them
    # This provides a measure of safety
    $self->db->cleanUp();

    $pm->start($file) and next;
    my ( $err, undef, $fh ) = $self->getReadFh($file);

    if ($err) {
      $self->log( 'fatal', $self->name . ": $err" );
    }

    my $versionLine = <$fh>;

    if ( !$versionLine ) {
      $self->log( 'fatal', $self->name . ": couldn't read version line of $file" );
    }

    chomp $versionLine;

    $self->log( 'debug', $self->name . ": read version line of $file: $versionLine" );

    if ( index( $versionLine, '## CADD' ) == -1 ) {
      $self->log( 'fatal',
        $self->name . ": first line of $file is not CADD formatted: $_" );
    }

    # Cadd's columns descriptor is on the 2nd line
    my $headerLine = <$fh>;

    if ( !$headerLine ) {
      $self->log( 'fatal', $self->name . ": couldn't read header line of $file" );
    }

    chomp $headerLine;

    $self->log( 'debug', $self->name . ": read header of $file: $headerLine" );

    # We may have converted the CADD file to a BED-like format, which has
    # chrom chromStart chromEnd instead of #Chrom Pos
    # and which is 0-based instead of 1 based
    # Moving $phastIdx to the last column
    my @headerFields = split '\t', $headerLine;

    # Get the last index, that's where the phast column lives https://ideone.com/zgtKuf
    # Can be 5th or 6th column idx. 5th for CADD file, 6th for BED-like file
    my $phastIdx = $#headerFields;

    my $altAlleleIdx = $#headerFields - 2;
    my $refBaseIdx   = $#headerFields - 3;

    my $based = $self->based;
    my $isBed;

    if ( @headerFields == 7 ) {
      # It's the bed-like format
      $based = 0;
      $isBed = 1;
    }

    $self->log( 'debug',
          $self->name
        . ": input file is "
        . ( $isBed ? "a bed-like file" : "a CADD (non-bed) file" ) );
    $self->log( 'debug', $self->name . ": input file is $based\-based" );

    # We accumulate information about why a record is bad
    my %skipSites;

    # Track which fields we recorded, to record in $self->completionMeta
    my %visitedChrs = ();

    # We keep track of posiitonal changes, to know when we're done accumulating scores
    # (provided that sorting is guaranteed)
    my $lastPosition;

    # the reference may not match up, report how many of these positions exist
    # this is very expected for lifted over files
    my $changedRefPositions   = 0;
    my $multiRefPositions     = 0;
    my $multiScorePositions   = 0;
    my $nonACTGrefPositions   = 0;
    my $nonACTGaltPositions   = 0;
    my $missingScorePositions = 0;

    # File does not need to be sorted by chromosome, but each position
    # must be in a block of 3 (one for each possible allele)
    my @fields;

    my ( $chr, $wantedChr, $dbPosition );
    my ( @caddData, $caddRef, $dbData, $assemblyRefBase, $altAllele, $refBase,
      $phredScoresAref );

    # Manage our own cursors, to improve performance
    my $cursor;
    my $count = 0;
    FH_LOOP: while ( my $line = $fh->getline() ) {
      chomp $line;

      @fields = split '\t', $line;

      # If this is a regular CADD file, it will not have "chr" prepended
      # Else it "should" be found in the beginning of the string
      # If not, it will be caught in our if( $self->chrIsWanted($chr) ) check
      # http://ideone.com/Y5PiUa

      # Normalizes the $chr representation to one we may want but did not specify
      # Example: 1 becomes chr1, and is checked against our list of wanted chromosomes
      # Avoids having to use a field transformation, since this may be very common
      # and Bystro typical use is with UCSC-style chromosomes
      # If the chromosome isn't wanted, $chr will be undefined
      $chr = $self->normalizedWantedChr->{ $fields[0] };

      #If the chromosome is new, write any data we have & see if we want new one
      if ( !defined $wantedChr || ( !defined $chr || $wantedChr ne $chr ) ) {
        # We switched chromosomes
        if ( defined $wantedChr ) {
          #Clean up the database, commit & close any cursors, free memory;
          $self->db->cleanUp();
          undef $cursor;

          #Reset our transaction counter
          $count = 0;

          if ( @caddData || defined $caddRef ) {
            my $err = $self->name . ": changed chromosomes, but unwritten data remained";

            $self->log( 'fatal', $err );
          }
        }

        # Completion meta checks to see whether this track is already recorded
        # as complete for the chromosome, for this track
        $wantedChr = $self->chrWantedAndIncomplete($chr);
        undef @caddData;
        undef $caddRef;
      }

      # We expect either one chr per file, or all in one file
      # However, chr-split CADD files may have multiple chromosomes after liftover
      # TODO: rethink chrPerFile handling
      if ( !defined $wantedChr ) {
        next FH_LOOP;
      }

      ### Record that we visited the chr, to enable recordCompletion later ###
      # //= is equivalent to checking for defined before assigning
      $visitedChrs{$wantedChr} //= 1;

      # CADD uses a number of IUPAC codes for multi reference sites, skip these
      if ( $fields[$refBaseIdx] ne 'A'
        && $fields[$refBaseIdx] ne 'C'
        && $fields[$refBaseIdx] ne 'G'
        && $fields[$refBaseIdx] ne 'T' )
      {
        $nonACTGrefPositions++;
        next FH_LOOP;
      }

      $dbPosition = $fields[1] - $based;

      ######## If we've changed position, we should have a 3 mer ########
      ####################### If so, write that ##############################
      ####################### If not, wait until we do #######################
      # Note, each call to dbPatch has a boolean !!($count >= $self->commitEvery)
      # because we delay commits to increase performance
      if ( defined $lastPosition && $lastPosition != $dbPosition ) {
        if ( defined $skipSites{"$wantedChr\_$lastPosition"} ) {
          #use debug because this logs hundreds of MB of data for lifted over hg38, and never seen a mistake
          #can have billions of messages, avoid string copy by checking if would log/print anything
          #From Seq::Role::Message
          if ( $self->hasDebugLevel ) {
            $self->log( 'debug',
                  $self->name
                . ": $wantedChr\:$lastPosition: "
                . $skipSites{"$wantedChr\_$lastPosition"}
                . ". Skipping" );
          }

          if ( @caddData || $caddRef ) {
            my $err =
              $self->name . ": skipSites and score accumulation should be mutually exclusive";
            $self->log( 'fatal', $err );
          }

          #Can delete because order guaranteed
          delete $skipSites{"$wantedChr\_$lastPosition"};
          # There is nothing to write in this case since sorting is guaranteed
        }
        elsif ( !@caddData ) {
          # Could occur if we skipped the lastPosition because refBase didn't match
          # assemblyRefBase
          $self->log( 'warn',
            $self->name . ": $wantedChr\:$lastPosition: No scores or warnings accumulated." );
          undef $caddRef;
        }
        else {
          $cursor //= $self->db->dbStartCursorTxn($wantedChr);

          ########### Check refBase against the assembly's reference #############
          # We read using our cursor; since in LMDB, cursors are isolated
          # and therefore don't want to use our helper dbRead class, as inconsistencies may arise
          $dbData          = $self->db->dbReadOneCursorUnsafe( $cursor, $lastPosition );
          $assemblyRefBase = $refTrack->get($dbData);

          if ( !defined $assemblyRefBase ) {
            my $err = $self->name . ": no assembly ref base found for $wantedChr:$lastPosition";
            $self->log( 'fatal', $err );
          }

          # When lifted over, reference base is not lifted, can cause mismatch
          # In these cases it makes no sense to store this position's CADD data
          if ( $assemblyRefBase ne $caddRef ) {
            # Don't log, in hg38 case there will be much of the genome logged
            $changedRefPositions++;

            #As long as sorting is guaranteed, there is no reason to write
            #anything in these cases
            undef @caddData;
            undef $caddRef;
          }
          else {
            $phredScoresAref =
              $self->_accumulateScores( $wantedChr, \@caddData, $caddRef, $lastPosition );

            if ( !defined $phredScoresAref ) {
              # Sorted guaranteed, but no score found
              # This can actually happen as a result of liftover
              # chr22 20668231 has 6 scores, because liftOver mapped 2 different positions
              # to 20668231, when lifting over from hg19
              if ( $self->hasDebugLevel ) {
                $self->log( 'debug',
                      $self->name
                    . ": $wantedChr\:$lastPosition: Instead of 3 scores got: "
                    . ( @caddData || 0 )
                    . ". Skipping" );
              }

              if ( @caddData > 3 ) {
                $multiScorePositions++;
              }
              else {
                my $err = $self->name
                  . ": $wantedChr\:$lastPosition: Didn't accumulate 3 phredScores, and not because > 3 scores, which should be impossible.";
                $self->log( 'fatal', $err );
                die $err;
              }

              #Since sorting is guaranteed, there is nothing to write here
            }
            else {
              #Args:                         $cursor               $chr,       $trackIndex,   $pos,         $trackValue
              $self->db->dbPatchCursorUnsafe( $cursor, $wantedChr, $self->dbName, $lastPosition,
                $phredScoresAref );

              if ( $count > $self->commitEvery ) {
                $self->db->dbEndCursorTxn($wantedChr);
                undef $cursor;

                $count = 0;
              }

              $count++;

              undef $phredScoresAref;
            }

            undef @caddData;
            undef $caddRef;
          }
        }
      }

      ##### Build up the scores into 3-mer (or 4-mer if ambiguous base) #####

      # This site will be next in 1 iteration
      $lastPosition = $dbPosition;

      if ( defined $skipSites{"$wantedChr\_$lastPosition"} ) {
        next;
      }

      $altAllele = $fields[$altAlleleIdx];
      $refBase   = $fields[$refBaseIdx];

      if ( $altAllele ne 'A'
        && $altAllele ne 'C'
        && $altAllele ne 'G'
        && $altAllele ne 'T' )
      {
        $skipSites{"$wantedChr\_$lastPosition"} = "non_actg_alt";
        $nonACTGaltPositions++;

        # No need to keep this in memory, since we never will use this value
        undef @caddData;
        undef $caddRef;
        next;
      }

      if ( $dbPosition < 0 ) {
        my $err = $self->name . ": found dbPosition < 0: $line. This is impossible.";
        $self->log( 'fatal', $err );
        die $err;
      }

      #if !defined $caddRef assign caddRef
      $caddRef //= $refBase;

      # If we find a position that has multiple bases, that is undefined behavior
      # so we will store a nil (undef on perl side, nil in msgpack) for cadd at that position
      # This can happen as result of liftover
      # This is NOT the same thing as the multiple base call that CADD sometimes
      # uses for the reference (e.g and "M", or "R")
      if ( $caddRef ne $refBase ) {
        # Mark for $missingValue insertion
        $skipSites{"$wantedChr\_$lastPosition"} = "multi_ref";
        $multiRefPositions++;

        # No need to keep this in memory, since we never will use this value
        undef @caddData;
        undef $caddRef;
        next;
      }

      # If no phastIdx found for this site, there cannot be 3 scores accumulated
      # so mark it as for skipping; important because when out of order
      # we may have cryptic 3-mers, which we don't want to insert
      if ( !defined $fields[$phastIdx] || !looks_like_number( $fields[$phastIdx] ) ) {
        # Mark for undef insertion
        $skipSites{"$wantedChr\_$lastPosition"} = "missing_score";
        $missingScorePositions++;

        # No need to keep this in memory, since we never will use this value
        undef @caddData;
        undef $caddRef;
        next;
      }

      push @caddData, [ $altAllele, $self->{_rounder}->round( $fields[$phastIdx] ) ];
    }

    ######################### Finished reading file ##########################
    ######### Collect any scores that were accumulated out of order ##########
    if (@caddData) {
      if ( !( defined $wantedChr && defined $lastPosition && defined $caddRef ) ) {
        my $err = $self->name
          . ": at end of file, if have cadd data, expect lastPosition, wantedChr, and cadRef";
        $self->log( 'fatal', $err );
        die $err;
      }

      if ( defined $cursor ) {
        $self->db->dbEndCursorTxn($wantedChr);
        undef $cursor;
      }

      if ( defined $skipSites{"$wantedChr\_$lastPosition"} ) {
        $self->log( 'debug',
              $self->name
            . ": $wantedChr\:$lastPosition: "
            . $skipSites{"$chr\_$lastPosition"}
            . ". Skipping." );

        # always safe to delete here; last time we'll check it
        delete $skipSites{"$wantedChr\_$lastPosition"};
        #Since sorting is guaranteed, no need to write anything
      }
      else {
        $dbData          = $self->db->dbReadOne( $wantedChr, $lastPosition );
        $assemblyRefBase = $refTrack->get($dbData);

        if ( $assemblyRefBase ne $caddRef ) {
          $changedRefPositions++;

          $self->log( 'debug',
            $self->name . ": $wantedChr\:$lastPosition: Ref doesn't match. Skipping." );
          #Since sorting is guaranteed, no need to write anything
        }
        else {
          $phredScoresAref =
            $self->_accumulateScores( $wantedChr, \@caddData, $caddRef, $lastPosition );

          # We want to keep missing values consistent
          # Because when sorting not guaranteed, we may want non-nil/undef
          # values to prevent cryptic 3-mers
          if ( !defined $phredScoresAref ) {
            $self->log( 'debug',
                  $self->name
                . ": $wantedChr\:$lastPosition: Instead of 3 scores got: "
                . ( @caddData || 0 )
                . ". Skipping" );

            if ( @caddData > 3 ) {
              $multiScorePositions++;
            }
            else {
              my $err = $self->name
                . ": $wantedChr\:$lastPosition: Didn't accumulate 3 phredScores, and not because > 3 scores, which should be impossible.";
              $self->log( 'fatal', $err );
            }
          }
          else {
            #We commit here, because we don't expect any more scores
            $self->db->dbPatch( $wantedChr, $self->dbName, $lastPosition, $phredScoresAref );
          }
        }
      }
    }

    undef @caddData;
    undef $lastPosition;
    undef $wantedChr;
    undef $phredScoresAref;

    #Commit any remaining transactions, commit & close cursors, sync all environments, free memory
    $self->db->cleanUp();
    undef $cursor;

    $self->safeCloseBuilderFh( $fh, $file, 'fatal' );

    if ( $changedRefPositions > 0 ) {
      $self->log( 'warn',
        $self->name
          . ": skipped $changedRefPositions positions because CADD Ref didn't match ours: in $file."
      );
    }

    if ( $multiRefPositions > 0 ) {
      $self->log( 'warn',
        $self->name
          . ": skipped $multiRefPositions positions because CADD Ref had multiple Ref at that position: in $file."
      );
    }

    if ( $multiScorePositions > 0 ) {
      $self->log( 'warn',
        $self->name
          . ": skipped $multiScorePositions positions because found multiple scores: in $file"
      );
    }

    if ( $nonACTGrefPositions > 0 ) {
      $self->log( 'warn',
        $self->name
          . ": skipped $nonACTGrefPositions positions because found non-ACTG CADD Ref: in $file"
      );
    }

    if ( $nonACTGaltPositions > 0 ) {
      $self->log( 'warn',
        $self->name
          . ": skipped $nonACTGaltPositions positions because found non-ACTG CADD Alt: in $file"
      );
    }

    if ( $missingScorePositions > 0 ) {
      $self->log( 'warn',
        $self->name
          . ": skipped $missingScorePositions positions because has missing Phred scores: in $file"
      );
    }

    $pm->finish( 0, \%visitedChrs );
  }

  $pm->wait_all_children();

  # Now, regardless of whether chromosomes were sorted, or spread across many files
  # we can record true completion state
  for my $chr ( keys %completedChrs ) {
    $self->completionMeta->recordCompletion($chr);

    $self->log( 'info',
          $self->name
        . ": recorded $chr completed, from "
        . ( join( ",", @{ $completedChrs{$chr} } ) ) );
  }

  #TODO: figure out why this is necessary, even with DEMOLISH
  $self->db->cleanUp();

  #TODO: Implement actual error return codes instead of dying
  return;
}

sub _accumulateScores {
  #my ($self, $chr, $dataAref, $caddRef, $lastPosition) = @_;
  #    $_[0] , $_[1], $_[2],   $_[3],    $_[4]

  # (@{$dataAref} != 3)
  if ( @{ $_[2] } != 3 ) {
    # May be called before 3 scores accumulated
    return undef;
  }

  # Found 3 scores
  # Make sure we place them in the correct order
  my @phredScores;
  my $index;
  #            ( @{$dataAref} )
  for my $aref ( @{ $_[2] } ) {
    #In the aref, first position is the altAllele, 2nd is the phred score
    #                   {$caddRef}
    $index = $order->{ $_[3] }{ $aref->[0] };

    # checks whether ref and alt allele are ACTG
    if ( !defined $index ) {
      my $err = $_[0]->name . ": $_[1]\:$_[4]: no score possible for altAllele $aref->[0]";
      #      $self->log
      $_[0]->log( 'fatal', $err );
    }

    $phredScores[$index] = $aref->[1];
  }

  return \@phredScores;
}

__PACKAGE__->meta->make_immutable;
1;
