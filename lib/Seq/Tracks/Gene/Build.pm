use 5.10.0;
use strict;
use warnings;

package Seq::Tracks::Gene::Build;

our $VERSION = '0.001';

# ABSTRACT: Builds Gene Tracks 
    # Stores refSeq data, nearest gene refSeq data, generates
    # in-silico transcribed transcripts (and associated fields)

    #Inserts a single value <ArrayRef> @ $self->name
    #If $self->nearest defined, inserts a <Int> @ $self->nearestFeatureName

use Mouse 2;
use namespace::autoclean;

use Parallel::ForkManager;

use Seq::Tracks::Gene::Build::TX;
use Seq::Tracks::Gene::Definition;
use Seq::Tracks;

extends 'Seq::Tracks::Build';
#exports regionTrackPath
with 'Seq::Tracks::Region::RegionTrackPath';

use DDP;
use List::Util qw/first/;

my $geneDef = Seq::Tracks::Gene::Definition->new();

# We don't remap field names
# It's easier to remember the real names than real names + our domain-specific names

#can be overwritten if needed in the config file, as described in Tracks::Build
has chrom_field_name => (is => 'ro', lazy => 1, default => 'chrom' );
has txStart_field_name => (is => 'ro', lazy => 1, default => 'txStart' );
has txEnd_field_name => (is => 'ro', lazy => 1, default => 'txEnd' );

has build_region_track_only => (is => 'ro', lazy => 1, default => 0);
has join => (is => 'ro', isa => 'HashRef');

# These are the features stored in the Gene track's region database
# Does not include $geneDef->txErrorName here, because that is something
# that is not actually present in UCSC refSeq or knownGene records, we add ourselves
has '+features' => (default => sub{ $geneDef->allUCSCgeneFeatures; });

my $txNumberKey = 'txNumber';
my $joinTrack;
sub BUILD {
  my $self = shift;

  # txErrorName isn't a default feature, initializing here to make sure 
  # we store this value (if calling for first time) before any threads get to it
  $self->getFieldDbName($geneDef->txErrorName);

  #similarly for $txSize
  # $self->getFieldDbName($geneDef->txSizeName);
}

# 1) Store a reference to the corresponding entry in the gene database (region database)
# 2) Store this codon information at some key, which the Tracks::Region::Gene
# 3) Store transcript errors, if any
# 4) Write region data
# 5) Write gene track data in main db
# 6) Write nearest genes if user wants those
sub buildTrack {
  my $self = shift;

  my @allFiles = $self->allLocalFiles;

  # Only allow 1 thread because perl eats memory like candy
  my $pm = Parallel::ForkManager->new($self->max_threads);

  if($self->join) {
    my $tracks = Seq::Tracks->new();
    $joinTrack = $tracks->getTrackBuilderByName($self->joinTrackName);
  }

  my %allIdx; # a map <Hash> { featureName => columnIndexInFile}
  my %regionIdx; #like allIdx, but only for features going into the region databae
  # Every row (besides header) describes a transcript
  my %allData;
  my %regionData;
  my %txStartData;

  my $wantedChr;
  my %txNumbers;
  my $allDataHref;
  my $txStart;
  my $txEnd;
  my $txNumber;

  # Assume one file per loop, or all sites in one file. Tracks::Build warns if not
  for my $file (@allFiles) {
    $pm->start($file) and next;
      my $fh = $self->get_read_fh($file);

      my $firstLine = <$fh>;

      # Fatal/exit will only affect that process, won't affect others
      if(!defined $firstLine) {
        my $err;

        if(!close($fh) && $? != 13) {
          $err = $self->name . ": $file failed to open due to: $! ($?)";
        } else {
          $err = $self->name . ": $file empty";
        }

        $self->log('fatal', $err);
        die $err;
      }

      chomp $firstLine;

      # Store all features we can find, for Seq::Build::Gene::TX. Avoid autocracy,
      # don't need to know what Gene::TX requires.
      my $fieldIdx = 0;
      for my $field (split '\t', $firstLine) {
        $allIdx{$field} = $fieldIdx;
        $fieldIdx++;
      }

      # Except w.r.t the chromosome field, txStart, txEnd, txNumber definitely need these
      if(!defined $allIdx{$self->chrom_field_name} || !defined $allIdx{$self->txStart_field_name}
      || !defined $allIdx{$self->txEnd_field_name} ) {
        my $err = $self->name . ": must provide chrom, txStart, txEnd fields";
        $self->log('fatal', $err);
        die $err;
      }
      # Region database features; as defined by user in the YAML config, or our default
      REGION_FEATS: for my $field ($self->allFeatureNames) {
        if(exists $allIdx{$field} ) {
          $regionIdx{$field} = $allIdx{$field};
          next REGION_FEATS;
        }

        my $err = $self->name . ": required $field missing in $file header: $firstLine";
        $self->log('fatal', $err);
        die $err;
      }

      my $skipped = 0;
      FH_LOOP: while (<$fh>) {
        chomp;
        my @fields = split('\t', $_);

        my $chr = $fields[ $allIdx{$self->chrom_field_name} ];

        # We may have already finished this chr, or may not have asked for it
        if( ($wantedChr && $wantedChr ne $chr) || !$wantedChr ) {
          $wantedChr = $self->chrIsWanted($chr) && $self->completionMeta->okToBuild($chr) ? $chr : undef;
        }

        if(!$wantedChr) {
          # if not wanted, and we have one chr per file, exit
          if($self->chrPerFile) {
            $self->log('info', $self->name . ": chrs in file $file not wanted or previously completed. Skipping");
            $skipped = 1;
            last FH_LOOP;
          }

          #not wanted, but multiple chr per file, skip
          next FH_LOOP;
        }

        # Keep track of our 0-indexed transcript refreence numbers
        if( !$txNumbers{$wantedChr} ) {
          $txNumbers{$wantedChr} = 0;
        }

        $txNumber = $txNumbers{$wantedChr};

        $allDataHref = {};

        my $fieldDbName;
        ACCUM_VALUES: for my $fieldName (keys %allIdx) {
          if($self->hasTransform($fieldName) ) {
            $fields[ $allIdx{$fieldName} ] = $self->transformField($fieldName, $fields[ $allIdx{$fieldName} ]);
          }

          my $data = $self->coerceFeatureType($fieldName, $fields[ $allIdx{$fieldName} ]);

          if(!defined $data) {
            next ACCUM_VALUES;
          }

          # if this is a field that we need to store in the region db
          # create a shortened field name
          $fieldDbName = $self->getFieldDbName($fieldName);

          $allDataHref->{$fieldName} = $data;

          if(!defined $regionIdx{$fieldName} ) {
            next ACCUM_VALUES;
          }

          #store under a shortened fieldName to save space in the db
          $regionData{$wantedChr}->{$txNumber}{ $fieldDbName } = $data;
        }

        $txStart = $allDataHref->{$self->txStart_field_name};

        if(!$txStart) {
          my $statement =  $self->name . ': missing transcript start ( we expected a value @ ' .
            $self->txStart_field_name . ')';

          $self->log('fatal', $statement);
          die $statement;
        }

        $txEnd = $allDataHref->{$self->txEnd_field_name};

        if(!$txEnd) {
          my $statement =  $self->name . ': missing transcript start ( we expected a value @ ' .
            $self->txEnd_field_name . ')';

          $self->log('fatal', $statement);
          die $statement;
        }

        #a field added by Bystro
        # $regionData{$wantedChr}->{$txNumber}{$self->getFieldDbName($geneDef->txSizeName)} = $txEnd + 1 - $txStart; 

        if(defined $txStartData{$wantedChr}{$txStart} ) {
          push @{ $txStartData{$wantedChr}{$txStart} }, [$txNumber, $txEnd];
        } else {
          $txStartData{$wantedChr}{$txStart} = [ [$txNumber, $txEnd] ];
        }

        $allDataHref->{$txNumberKey} = $txNumber;

        push @{ $allData{$wantedChr}{$txStart} }, $allDataHref;

        $txNumbers{$wantedChr} += 1;
      }

      # If we fork a process in order to read (example zcat) prevent that process
      # from becoming defunct
      if(!close($fh) && $? != 13) {
        my $err = $self->name . ": failed to close $file due to $! ($?)";
        $self->log('fatal', $err);
        die $err;
      } else {
        $self->log('info', $self->name . ": closed $file with $?");
      }

      if($skipped) {
        # This returns to parent, in $pm->run_on_finish
        $pm->finish(0);
      }

      # If we skipped the $file, will never get here, so this is an error
      if(!%allData) {
        my $err = $self->name . ": no transcript data accumulated";
        $self->log('fatal', $err);
        die $err;
      }

      ############################### Make transcripts #########################
      my @allChrs = keys %allData;

      my $txErrorDbname = $self->getFieldDbName($geneDef->txErrorName);

      my $mainMergeFunc = sub {
        # Only when called when there is a defined $oldVal
        my ($chr, $pos, $oldVal, $newVal) = @_;
        # make it an array of arrays (array of geneTrack site records)
        if(!ref $oldVal->[0]) {
          return (undef, [$oldVal, $newVal]);
        }

        #oldVal is an array of arrays, push on to it
        my @updatedVal = @$oldVal;

        push @updatedVal, $newVal;

        #TODO: Should we throw any errors?
        return (undef, \@updatedVal);
      };

      TX_LOOP: for my $chr (@allChrs) {
        # We may want to just update the region track,
        # TODO: Note that this won't store any txErrors
        if($self->build_region_track_only) {
          $self->_writeRegionData( $chr, $regionData{$chr});

          if($self->join) {
            $self->_joinTracksToGeneTrackRegionDb($chr, $txStartData{$chr} );
          }

          next TX_LOOP;
        }

        $self->log('info', $self->name . ": starting to build transcripts for $chr");

        my @allTxStartsAscending = sort { $a <=> $b } keys %{ $allData{$chr} };

        my ($txInfo, $txNumber);
        for my $txStart ( @allTxStartsAscending ) {
          for my $txData ( @{ $allData{$chr}->{$txStart} } ) {
            $txNumber = $txData->{$txNumberKey};

            $txInfo = Seq::Tracks::Gene::Build::TX->new($txData);

            if(@{$txInfo->transcriptSites} %2 != 0) {
              my $err = $self->name . ": expected txSiteDataAndPos to contain (position1, value1, position2, value2) data";
              $self->log('fatal', $err);
              die $err;
            }

            my $pos;
            INNER: for (my $i = 0; $i < @{$txInfo->transcriptSites}; $i += 2) {
              $pos = $txInfo->transcriptSites->[$i];
              # $i corresponds to $pos, $i + 1 to value
              # Commit for every position
              # This also ensures that $mainMergeFunc will always be called with fresh data
              $self->db->dbPatch(
                $chr,
                $self->dbName,
                $pos,
                #value <Array[Scalar]>
                $txInfo->transcriptSites->[$i + 1],
                #how we handle cases where multiple overlap
                $mainMergeFunc
              );
            }

            if( @{$txInfo->transcriptErrors} ) {
              my @errors = @{$txInfo->transcriptErrors};
              $regionData{$chr}->{$txNumber}{$txErrorDbname} = \@errors;
            }
          }

          delete $allData{$chr}->{$txStart};
        }

        delete $allData{$chr};

        # We map a lot of memory by this point, so release it back to the OS
        # To reduce likelihood that linux will want to swap
        # The database will be re-opened as needed
        $self->db->cleanUp($chr);

        $self->log('info', $self->name . ": finished building transcripts for $chr");

        $self->_writeRegionData($chr, $regionData{$chr});

        delete $regionData{$chr};

        if($self->join) {
          $self->_joinTracksToGeneTrackRegionDb($chr, $txStartData{$chr} );
        }

        # For gene tracks we require features
        if(!$self->noNearestFeatures) {
          $self->_writeNearestGenes($chr, $txStartData{$chr});
        }

        delete $txStartData{$chr};

        #Commit, sync everything, including completion status, and release mmap
        $self->db->cleanUp($chr);

         # We've finished with 1 chromosome, so write that to meta to disk
         # TODO: error check this
        $self->completionMeta->recordCompletion($chr);

        $self->log('info', $self->name . ": recorded $chr completed");
      }

      #Commit, sync everything, including completion status, and release mmap
      $self->db->cleanUp();
    $pm->finish(0);
  }

  $pm->run_on_finish( sub {
    my ($pid, $exitCode, $fileName, $exitSignal, $coreDump) = @_;

    if($exitCode != 0) {
      my $err = $self->name . ": got exitCode $exitCode for $fileName: $exitSignal . Dump: $coreDump";

      $self->log('fatal', $err);
      die $err;
    }

    #Only message that is different, in that we don't pass the $fileName
    $self->log('info', $self->name . ": completed building from $fileName");
  });

  $pm->wait_all_children;
  return;
}

sub _writeRegionData {
  my ($self, $chr, $regionDataHref) = @_;

  $self->log('info', $self->name . ": starting _writeRegionData for $chr");

  my $dbName = $self->regionTrackPath($chr);

  my @txNumbers = keys %$regionDataHref;

  for my $txNumber (@txNumbers) {
    # Patch one at a time, because we assume performance isn't an issue
    # And neither is size, so hash keys are fine
    $self->db->dbPatchHash($dbName, $txNumber, $regionDataHref->{$txNumber});
  }

  $self->log('info', $self->name . ": finished _writeRegionData for $chr");
}

############ Joining some other track to Gene track's region db ################

my $tracks = Seq::Tracks->new();

sub _joinTracksToGeneTrackRegionDb {
  my ($self, $chr, $txStartData) = @_;

  if(!$self->join) {
    return $self->log('warn', $self->name . ": join not set in _joinTracksToGeneTrackRegionDb");
  }

  $self->log('info', $self->name . ": starting _joinTracksToGeneTrackRegionDb for $chr");
  # Gene tracks cover certain positions, record the start and stop
  my @positionRanges;
  my @txNumbers;

  for my $txStart (keys %$txStartData) {
    foreach ( @{ $txStartData->{$txStart} } ) {
      my $txNumber = $_->[0];
      my $txEnd = $_->[1];
      push @positionRanges, [ $txStart, $txEnd ];
      push @txNumbers, $txNumber;
    }
  }

  # TODO: Add check to see if values have already been entered
  my $mergeFunc = sub {
    my ($chr, $pos, $oldVal, $newVal) = @_;

    my @updated;

    #If the old value is an array, push the new values on to the old values
    if(ref $oldVal) {
      @updated = @$oldVal;

      for my $val (ref $newVal ? @$newVal : $newVal) {
        if(!defined $val) {
          next;
        }

        push @updated, $val;
      }
    } else {
      if(defined $oldVal) {
        @updated = ($oldVal);
      }

      for my $val (ref $newVal ? @$newVal : $newVal) {
        if(!defined $val) {
          next;
        }

        # If not array I want to see an error
        push @updated, $val;
      }
    }

    # Try to add as little junk as possible
    if(@updated == 0) {
      return (undef, $oldVal);
    }

    if(@updated == 1) {
      return (undef, $updated[0]);
    }

    return (undef, \@updated);
  };

  my $dbName = $self->regionTrackPath($chr);

  # For each txNumber, run dbPatchHash on any joining data
  $joinTrack->joinTrack($chr, \@positionRanges, $self->joinTrackFeatures, sub {
    # Called every time a match is found
    # Index is the index of @ranges that this update belongs to
    my ($hrefToAdd, $index) = @_;

    my %out;
    foreach (keys %$hrefToAdd) {
      if(defined $hrefToAdd->{$_}) {
        if(ref $hrefToAdd->{$_} eq 'ARRAY') {
          my @arr;
          my %uniq;
          for my $entry (@{$hrefToAdd->{$_}}) {
            if(defined $entry) {
              if(!$uniq{$entry}) {
                push @arr, $entry;
              }
              $uniq{$entry} = 1;
            }
          }

          # Don't add empty arrays to the database
          $hrefToAdd->{$_} = @arr ? \@arr : undef;
        }

        if(defined $hrefToAdd->{$_}) {
          # Our LMDB writer requires a value, so only add to our list of db entries
          # to update if we have a value
          #$self->getFieldDbName generates a name for the field we're joining, named $_
          $self->db->dbPatchHash($dbName, $txNumbers[$index], {
            $self->getFieldDbName($_) => $hrefToAdd->{$_}
          }, $mergeFunc);
        }
      }
    }

    # Free memory as soon as possible
    undef $hrefToAdd;
    undef %out;
  });

  $self->log('info', $self->name . ": finished _joinTracksToGeneTrackRegionDb for $chr");
}

############### Writing nearest gene data to main database #####################

# Find all of the nearest genes, for any intergenic regions
# Genic regions by our definition are nearest to themselves
# All UCSC refGene data is 0-based
# http://www.noncode.org/cgi-bin/hgTables?db=hg19&hgta_group=genes&hgta_track=refGene&hgta_table=refGene&hgta_doSchema=describe+table+schema
sub _writeNearestGenes {
  my ($self, $chr, $txStartData) = @_;

  $self->log('info', $self->name . ": starting _writeNearestGenes for $chr");

  # Get database length : assumes reference track already in the db
  my $genomeNumberOfEntries = $self->db->dbGetNumberOfEntries($chr);

  my @allTranscriptStarts = sort { $a <=> $b } keys %$txStartData;

  # Track the longest (further in db toward end of genome) txEnd, because
  #  in  case of overlapping transcripts, want the points that ARENT 
  #  covered by a gene (since those have apriori nearest records: themselves)
  #  This also acts as our starting position
  my $longestPreviousTxEnd = 0;
  my $longestPreviousTxNumbers;

  my ($txStart, $txNumber, $midPoint, $posTxNumber, $previousTxStart);

  my $count = 0;
  TXSTART_LOOP: for (my $n = 0; $n < @allTranscriptStarts; $n++) {
    $txStart = $allTranscriptStarts[$n];

    # If > 1 transcript shares a start, txNumber will be an array of numbers
    # <ArrayRef[Int]> of length 1 or more
    # else will be a scalar, save some space in db, and reduce Perl memory growth
    $txNumber =
      @{$txStartData->{$txStart}} > 1
      ? [ map { $_->[0] } @{ $txStartData->{$txStart} } ]
      : $txStartData->{$txStart}[0][0];

    if($n > 0) {
      # Look over the upstream txStart, see if it overlaps
      # We take into account the history of previousTxEnd's, for non-adjacent
      # overlapping transcripts
      $previousTxStart = $allTranscriptStarts[$n - 1];

      for my $txItem ( @{ $txStartData->{$previousTxStart} } ) {
        if($txItem->[1] > $longestPreviousTxEnd) {
          $longestPreviousTxEnd =  $txItem->[1];

          $longestPreviousTxNumbers = $txItem->[0];
          next;
        }

        if($txItem->[1] == $longestPreviousTxEnd) {
          if(!ref $longestPreviousTxNumbers) {
            $longestPreviousTxNumbers = [$longestPreviousTxNumbers];
          }

          push @$longestPreviousTxNumbers, $txItem->[0];
        }
      }

      # Take the midpoint of the longestPreviousTxEnd .. txStart - 1 region
      $midPoint = $longestPreviousTxEnd + ( ( ($txStart - 1) - $longestPreviousTxEnd ) / 2 );
    }

    #### Accumulate txNumber or longestPreviousTxNumber for positions between transcripts #### 

    # When true, we are not intergenic
    if($longestPreviousTxEnd < $txStart) {
      # txEnd is open, 1-based so include, txStart is closed, 0-based, so stop 1 base before it
      POS_LOOP: for my $pos ( $longestPreviousTxEnd .. $txStart - 1 ) {
        if($n == 0 || $pos >= $midPoint) {
          #Args:             $chr,       $trackIndex,   $pos,  $trackValue, $mergeFunc, $skipCommit
          $self->db->dbPatch($chr, $self->nearestDbName, $pos, $txNumber, undef, $count < $self->commitEvery);
        } else {
          #Args:             $chr,       $trackIndex,   $pos,  $trackValue,             $mergeFunc, $skipCommit
          $self->db->dbPatch($chr, $self->nearestDbName, $pos, $longestPreviousTxNumbers, undef, $count < $self->commitEvery);
        }

        $count = $count < $self->commitEvery ? $count + 1 : 0;
      }
    }

    #Just in case, force commit between these two sections
    $self->db->dbForceCommit($chr);
    $count = 0;

    ###### Accumulate txNumber or longestPreviousTxNumber for positions after last transcript in the chr ######
    if ($n == @allTranscriptStarts - 1) {
      my $nearestNumber;
      my $startPoint;

      #maddingly perl reduce doesn't seem to work, despite this being an array
      my $longestTxEnd = 0;
      foreach (@{ $txStartData->{$txStart} }) {
        $longestTxEnd = $longestTxEnd > $_->[1] ? $longestTxEnd : $_->[1];
      }

      if($longestTxEnd > $longestPreviousTxEnd) {
        $nearestNumber = $txNumber;

        $startPoint = $longestTxEnd;
      } elsif ($longestTxEnd == $longestPreviousTxEnd) {
        $nearestNumber = [
          ref $longestPreviousTxNumbers ? @$longestPreviousTxNumbers : $longestPreviousTxNumbers,
          ref $txNumber ? @$txNumber : $txNumber
        ];

        $startPoint = $longestTxEnd;
      } else {
        $nearestNumber = $longestPreviousTxNumbers;

        $startPoint = $longestPreviousTxEnd;
      }

      if($self->hasDebugLevel) {
        say "genome last position is @{[$genomeNumberOfEntries-1]}";
        say "longestTxEnd is $longestTxEnd";
        say "longestPreviousTxEnd is $longestPreviousTxEnd";
        say "current end > previous? " . ($longestTxEnd > $longestPreviousTxEnd ? "YES" : "NO");
        say "previous end equal current? " . ($longestTxEnd == $longestPreviousTxEnd ? "YES" : "NO");
        say "nearestNumber is";
        p $nearestNumber;
        say "starting point in last is $startPoint";
      }

      END_LOOP: for my $pos ( $startPoint .. $genomeNumberOfEntries - 1 ) {
        #Args:             $chr,       $trackIndex,   $pos,  $trackValue,   $mergeFunc, $skipCommit
        $self->db->dbPatch($chr, $self->nearestDbName, $pos, $nearestNumber, undef, $count < $self->commitEvery);
        $count = $count < $self->commitEvery ? $count + 1 : 0;
      }
    }
  }

  $self->log('info', $self->name . ": finished _writeNearestGenes for $chr");
}

__PACKAGE__->meta->make_immutable;
1;