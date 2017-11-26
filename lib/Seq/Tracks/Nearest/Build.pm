use 5.10.0;
use strict;
use warnings;

package Seq::Tracks::Nearest::Build;

our $VERSION = '0.001';

# ABSTRACT: Builds Gene Tracks 
    # Stores refSeq data, nearest gene refSeq data, generates
    # in-silico transcribed transcripts (and associated fields)

    #Inserts a single value <ArrayRef> @ $self->name
    #If $self->nearest defined, inserts a <Int> @ $self->nearestFeatureName

use Mouse 2;
use namespace::autoclean;

use Parallel::ForkManager;
use Scalar::Util qw/looks_like_number/;
use DDP;
use List::Util qw/max min uniq/;
use Digest::MD5 qw/md5/;

use Seq::Tracks;

extends 'Seq::Tracks::Build';
#exports regionTrackPath
with 'Seq::Tracks::Region::RegionTrackPath';

# TODO: Currently we mutate the 'from' and 'to' properties
# such that these may not keep 1:1 correspondance in case of overlapping transcripts
# To get around this we could store shadow properties of these
# for distance calculation purposes

# Coordinate we start looking from
# Typically used for gene tracks, would be 'txStart'
# This, at the moment, must be a column in which each value is a number
# So for instance, exonEnds, which have multiple comma-separated values
# Wouldn't work, since they would either be treated as a string,
# or through build_field_transformations: exonEnds: split(,)
# would appear to this program as an array of numbers, which this program
# doesn't currently know what to do with
has from => (is => 'ro', isa => 'Str', required => 1);

# Coordinate we look to
# Typically used for gene tracks, would 'txEnd' or nothing
# Similarly, should be a column of numbers
# Not required, we may opt to check against a single point, like txStart, for
# a "nearestTss" track
# To simplify the funcitons, we default to the "from" attribute
# In our BUILDARGS
has to => (is => 'ro', isa => 'Str', lazy => 1, default => sub {
  my $self = shift;
  return $self->from;
});

# If we're not given local_files, get them from a track reference
has ref => (is => 'ro', isa => 'Seq::Tracks::Build');

# So that we know which db to store within (db is segregated on chromosome)
has chromField => (is => 'ro', isa => 'Str', default => 'chrom');

my $txNumberKey = 'txNumber';

around BUILDARGS => sub {
  my($orig, $self, $data) = @_;

  p $data;
  if(!defined $data->{local_files}) {
    # Careful with the reference
    $data->{local_files} = $data->{ref}->local_files;
  }

  # We require non-empty 'from'
  if(!$data->{from}) {
    # A nicer exit
    $self->log('fatal', "'from' property must be specified for 'nearest' tracks");
  }

  ############# Add "from" and "to" to "features" if not present ###############
  # To allow distance calculations in the getter (Nearest.pm)
  $data->{features} //= [];

  my $hasFrom;
  my $hasTo;
  for my $feat (@{$data->{features}}) {
    if($feat eq $data->{from}) {
      $hasFrom = 1;
    }

    if($data->{to} && $feat eq $data->{to}) {
      $hasTo = 1;
    }
  }

  if($data->{to} && !$hasTo) {
    unshift @{$data->{features}}, $data->{to};
  }

  if(!$hasFrom) {
    unshift @{$data->{features}}, $data->{from};
  }

  ##### Ensure that we get the exact 0-based, half-open coordinates correct ######
  # allows us to consider from .. to rather than from .. to - 1, from - 1 .. to, etc
  # TODO: don't assume UCSC-style genes, and require a build_field_transformation
  # For anything other than 'from' as 0-based closed and 'to' as 0-based open (+1 of true)
  if($data->{build_field_transformations}{$data->{from}} && (!$hasTo || $data->{build_field_transformations}{$data->{to}})) {
    return $self->$orig($data);
  }

  # We softly enforce build_field_transformations for the comm
  if($data->{from} eq 'txEnd' || $data->{from} eq 'cdsEnd') {
    $data->{build_field_transformations}{$data->{from}} = '- 1';
  }

  if($hasTo && ($data->{to} eq 'txEnd' || $data->{to} eq 'cdsEnd')) {
    $data->{build_field_transformations}{$data->{to}} = '- 1';
  }

  return $self->$orig($data);
};

sub BUILD {
  my $self = shift;

  # We require these two fields, so make sure we make db names for them
  # These are stored in the region database, allowing us to calculate distnace,
  # Should that be needed
  # We'll store these regardless of the 'dist' property
  # Won't add many bytes (<18 per transcript), and add flexibility for the user
  $self->getFieldDbName($self->from);
  $self->getFieldDbName($self->to);

  #Check that 
}

# Find all of the nearest genes, for any intergenic regions
# Genic regions by our definition are nearest to themselves
# All UCSC refGene data is 0-based
# http://www.noncode.org/cgi-bin/hgTables?db=hg19&hgta_group=genes&hgta_track=refGene&hgta_table=refGene&hgta_doSchema=describe+table+schema
sub buildTrack {
  my $self = shift;

  my @allFiles = $self->allLocalFiles;

  # Only allow 1 thread because perl eats memory like candy
  my $pm = Parallel::ForkManager->new($self->max_threads);

  my %allIdx; # a map <Hash> { featureName => columnIndexInFile}
  my %regionIdx; #like allIdx, but only for features going into the region databae
  # Every row (besides header) describes a transcript
  my %regionData;

  # We organize our data by the "from" position, to simplify look
  my %txStartData;

  my $wantedChr;
  my %txNumbers;
  my $allDataHref;
  my $from;
  my $to;
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
      }

      chomp $firstLine;

      # Store all features we can find, for Seq::Build::Gene::TX. Avoid autocracy,
      # don't need to know what Gene::TX requires.
      my $fieldIdx = 0;
      for my $field (split '\t', $firstLine) {
        $allIdx{$field} = $fieldIdx;
        $fieldIdx++;
      }

      my $fromIdx = $allIdx{$self->from};
      my $toIdx = $allIdx{$self->to};

      # Except w.r.t the chromosome field, txStart, txEnd, txNumber definitely need these
      if( !(defined $allIdx{$self->chromField} && defined $fromIdx && defined $toIdx) ) {
        $self->log('fatal', $self->name . ': must provide chrom, from, to fields');
      }

      # Region database features; as defined by user in the YAML config, or our default
      REGION_FEATS: for my $field (@{$self->features}) {
        if(exists $allIdx{$field} ) {
          $regionIdx{$field} = $allIdx{$field};
          next REGION_FEATS;
        }

        $self->log('fatal', $self->name . ": required $field missing in $file header: $firstLine");
      }

      # We add the "from" and "to" fields to allow distance calculations

      # Read the file.
      # We store everything on the basis of chr, so that we can accept
      # Either a file that contains multiple chromosomes
      # Or multiples file that contains a single chromosome each
      my $skipped = 0;
      my $fromDbName = $self->getFieldDbName($self->from);
      my $toDbName = $self->getFieldDbName($self->to);
      my $rowIdx = 0;
      FH_LOOP: while (<$fh>) {
        chomp;
        my @fields = split('\t', $_);

        my $chr = $fields[ $allIdx{$self->chromField} ];

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

        my %rowData;
        ACCUM_VALUES: for my $fieldName (keys %regionIdx) {
          my $data = $fields[ $regionIdx{$fieldName} ];

          # say split, etc; comes first so that each individual value
          # in an array (if split) can be coerced
          if($self->hasTransform($fieldName) ) {
            $data= $self->transformField($fieldName, $data);
          }

          # convert the value into some type, typically number(N)
          $data = $self->coerceFeatureType($fieldName, $data);

          # if this is a field that we need to store in the region db
          # create a shortened field name
          my $fieldDbName = $self->getFieldDbName($fieldName);

          #store under a shortened fieldName to save space in the db
          $rowData{ $fieldDbName } = $data;
        }

        my $from = $rowData{$fromDbName};
        my $to = $rowData{$toDbName};

        if( !(defined $from && defined $to && looks_like_number($from) && looks_like_number($to)) ) {
          $self->log('fatal', "Expected numeric 'from' and 'to' fields, found: $from and $to");
        }

        $regionData{$wantedChr}{$rowIdx} = [$rowIdx, \%rowData];

        $rowIdx++;
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
        $pm->finish(0); #returns to parent process here
      }

      # We've now accumulated everything from this file
      # So write it. LMDB will serialize writes, so this is fine, even 
      # if the file is not properly organized by chromosome
      for my $chr (keys %regionData) {
        # say "called $chr";
        my $compactRegionData = $self->_makeRegionData($regionData{$chr});
        p $compactRegionData;
        # $self->_writeRegionData($chr, $compactRegionData);

        # $self->_writeNearest($chr, $txStartData{$chr});

        #  # We've finished with 1 chromosome, so write that to meta to disk
        #  # TODO: error check this
        # $self->completionMeta->recordCompletion($chr);

        # $self->log('info', $self->name . ": recorded $chr completed");
      }

      #Commit, sync everything, including completion status, and release mmap
      $self->db->cleanUp();
    $pm->finish(0);
  }

  $pm->wait_all_children;
  return;
}

sub _makeRegionData {
  my ($self, $regionDataHref) = @_;

  my $fromDbName = $self->getFieldDbName($self->from);
  my $toDbName = $self->getFieldDbName($self->to);

  my @sorted = sort { $a->[1]{$fromDbName} <=> $b->[1]{$fromDbName} } values %{$regionDataHref};

  # my $minFrom = map {}
  # p @sorted;
  # exit;
  my @featureKeys = sort { $a <=> $b } keys %{$sorted[0][1]};

  my %regionData;
  my $txNumber = 0;
  my $count = 0;
  my %completed;
  my %unique;
  my %overlappingRegions;
  my $from;
  my $to;
  while(@sorted > 0) {
    # say "length of sorted was " . (scalar @sorted);
    my $row = shift @sorted;

    $from = $row->[1]{$fromDbName};
    $to = $row->[1]{$toDbName};

    say "processing id $row->[0] ($from - $to)";

    for my $pos ($from .. $to) {
      if($completed{$pos}) {
        next;
      }

      my @ids = ($row->[0]);

      I_LOOP: for my $iRow (@sorted) {
        my $iFrom = $iRow->[1]{$fromDbName};
        my $iTo = $iRow->[1]{$toDbName};

        if($pos >= $iFrom && $pos <= $iTo) {
          push @ids, $iRow->[0];
        } elsif($iFrom > $pos) {
          last I_LOOP;
        }
      }

      # Sort should not be needed, but to be safe
      my $id = join('_', sort { $a <=> $b } @ids);

      if(!exists $unique{$id}) {
        # p $ids;
        $regionData{$txNumber} = \@ids;
        $unique{$id} = $txNumber;
        $txNumber++;
      }

      $completed{$pos} = 1;

      # Assign the transcript number
      $overlappingRegions{$pos} = $unique{$id};
    }
  }

  for my $regionIds (values %regionData) {
    my @values;
    $#values = $featureKeys[-1];

    my %uniqueRows;
    for my $id (@$regionIds) {
      p $regionDataHref->{$id}[1];
      # De-duplicate records, while retaining any unique relationships
      # It often happens, say with UCSC refGene data, that even across all
      # desired features, rows are not unique
      # Check across all values other than from and to, these are updated
      # to be the widest interval across all shared positions
      # so at this time, there is no sense in checking uniqueness inclusive
      # of from and to
      my @nonFromTo;

      for my $key (@featureKeys) {
        if($key != $fromDbName && $key != $toDbName) {
          push @nonFromTo, $regionDataHref->{$id}[1]{$key};
        }
      }

      my $hash = md5(@nonFromTo);
      
      if($uniqueRows{$hash}) {
        next;
      }

      for my $intKey (@featureKeys) {
        push @{$values[$intKey]}, $regionDataHref->{$id}[1]{$intKey};
      }

      $uniqueRows{$hash} = 1;
    }

    for my $intKey (@featureKeys) {
      if(@{$values[$intKey]} == 1) {
        $values[$intKey] = $values[$intKey][0];
      }
    }

    if(ref $values[$fromDbName]) {
      $values[$fromDbName] = uniq(@{$values[$fromDbName]});
      $values[$toDbName] = uniq(@{$values[$toDbName]});
    }

    # Set the $regionData{$txNumber} value to \@values;
    $regionIds = \@values;
  }

  return \%regionData;
  # p %overlappingRegions;
  # Accumulate overlapping region data for each position from .. to
  # each overlapping pair gets a number
  # Store that number, write that number => [values] to region db
  # and return the hash of number => [values]

  # for my $
  # if($self->from eq $self->to) {
  #   return $self->_make1dRegionData($regionDataHref);
  # }

  # return $self->_make2dRegionData($regionDataHref);
  # Find completely overlapping transcript sets
  # These will most typically overlap
}

# sub _make1dRegionData {
#   my ($self, $regionDataHref) = @_;
# }

sub _writeRegionData {
  # my ($self, $chr, $regionDataHref) = @_;

  # $self->log('info', $self->name . ": starting _writeRegionData for $chr");

  # my $dbName = $self->regionTrackPath($chr);

  # my @txNumbers = keys %$regionDataHref;

  # for my $txNumber (@txNumbers) {
  #   # Patch one at a time, because we assume performance isn't an issue
  #   # And neither is size, so hash keys are fine
  #   $self->db->dbPatchHash($dbName, $txNumber, $regionDataHref->{$txNumber});
  # }

  # $self->log('info', $self->name . ": finished _writeRegionData for $chr");
}

# TODO : combine _writeNearestFrom and _writeNearestFromTo
sub _writeNearestFrom {
  # my ($self, $chr, $txStartData) = @_;

  # $self->log('info', $self->name . ": starting _writeNearestGenes for $chr");

  # # Get database length : assumes reference track already in the db
  # my $genomeNumberOfEntries = $self->db->dbGetNumberOfEntries($chr);

  # my @allTranscriptStarts = sort { $a <=> $b } keys %$txStartData;

  # # Track the longest (further in db toward end of genome) txEnd, because
  # #  in  case of overlapping transcripts, want the points that ARENT 
  # #  covered by a gene (since those have apriori nearest records: themselves)
  # #  This also acts as our starting position
  # my $longestPreviousTxEnd = 0;
  # my $longestPreviousTxNumbers;

  # my ($txStart, $txNumber, $midPoint, $posTxNumber, $previousTxStart);

  # my $count = 0;
  # TXSTART_LOOP: for (my $n = 0; $n < @allTranscriptStarts; $n++) {
  #   $txStart = $allTranscriptStarts[$n];

  #   # If > 1 transcript shares a start, txNumber will be an array of numbers
  #   # <ArrayRef[Int]> of length 1 or more
  #   # else will be a scalar, save some space in db, and reduce Perl memory growth
  #   $txNumber =
  #     @{$txStartData->{$txStart}} > 1
  #     ? [ map { $_->[0] } @{ $txStartData->{$txStart} } ]
  #     : $txStartData->{$txStart}[0][0];

  #   if($n > 0) {
  #     # Look over the upstream txStart, see if it overlaps
  #     # We take into account the history of previousTxEnd's, for non-adjacent
  #     # overlapping transcripts
  #     $previousTxStart = $allTranscriptStarts[$n - 1];

  #     for my $txItem ( @{ $txStartData->{$previousTxStart} } ) {
  #       if($txItem->[1] > $longestPreviousTxEnd) {
  #         $longestPreviousTxEnd =  $txItem->[1];

  #         $longestPreviousTxNumbers = $txItem->[0];
  #         next;
  #       }

  #       if($txItem->[1] == $longestPreviousTxEnd) {
  #         if(!ref $longestPreviousTxNumbers) {
  #           $longestPreviousTxNumbers = [$longestPreviousTxNumbers];
  #         }

  #         push @$longestPreviousTxNumbers, $txItem->[0];
  #       }
  #     }

  #     # Take the midpoint of the longestPreviousTxEnd .. txStart - 1 region
  #     $midPoint = $longestPreviousTxEnd + ( ( ($txStart - 1) - $longestPreviousTxEnd ) / 2 );
  #   }

  #   #### Accumulate txNumber or longestPreviousTxNumber for positions between transcripts #### 

  #   # When true, we are not intergenic
  #   if($longestPreviousTxEnd < $txStart) {
  #     # txEnd is open, 1-based so include, txStart is closed, 0-based, so stop 1 base before it
  #     POS_LOOP: for my $pos ( $longestPreviousTxEnd .. $txStart - 1 ) {
  #       if($n == 0 || $pos >= $midPoint) {
  #         #Args:             $chr,       $trackIndex,   $pos,  $trackValue, $mergeFunc, $skipCommit
  #         $self->db->dbPatch($chr, $self->dbName, $pos, $txNumber, undef, $count < $self->commitEvery);
  #       } else {
  #         #Args:             $chr,       $trackIndex,   $pos,  $trackValue,             $mergeFunc, $skipCommit
  #         $self->db->dbPatch($chr, $self->dbName, $pos, $longestPreviousTxNumbers, undef, $count < $self->commitEvery);
  #       }

  #       $count = $count < $self->commitEvery ? $count + 1 : 0;
  #     }
  #   }

  #   #Just in case, force commit between these two sections
  #   $self->db->dbForceCommit($chr);
  #   $count = 0;

  #   ###### Accumulate txNumber or longestPreviousTxNumber for positions after last transcript in the chr ######
  #   if ($n == @allTranscriptStarts - 1) {
  #     my $nearestNumber;
  #     my $startPoint;

  #     #maddingly perl reduce doesn't seem to work, despite this being an array
  #     my $longestTxEnd = 0;
  #     foreach (@{ $txStartData->{$txStart} }) {
  #       $longestTxEnd = $longestTxEnd > $_->[1] ? $longestTxEnd : $_->[1];
  #     }

  #     if($longestTxEnd > $longestPreviousTxEnd) {
  #       $nearestNumber = $txNumber;

  #       $startPoint = $longestTxEnd;
  #     } elsif ($longestTxEnd == $longestPreviousTxEnd) {
  #       $nearestNumber = [
  #         ref $longestPreviousTxNumbers ? @$longestPreviousTxNumbers : $longestPreviousTxNumbers,
  #         ref $txNumber ? @$txNumber : $txNumber
  #       ];

  #       $startPoint = $longestTxEnd;
  #     } else {
  #       $nearestNumber = $longestPreviousTxNumbers;

  #       $startPoint = $longestPreviousTxEnd;
  #     }

  #     if($self->hasDebugLevel) {
  #       say "genome last position is @{[$genomeNumberOfEntries-1]}";
  #       say "longestTxEnd is $longestTxEnd";
  #       say "longestPreviousTxEnd is $longestPreviousTxEnd";
  #       say "current end > previous? " . ($longestTxEnd > $longestPreviousTxEnd ? "YES" : "NO");
  #       say "previous end equal current? " . ($longestTxEnd == $longestPreviousTxEnd ? "YES" : "NO");
  #       say "nearestNumber is";
  #       p $nearestNumber;
  #       say "starting point in last is $startPoint";
  #     }

  #     END_LOOP: for my $pos ( $startPoint .. $genomeNumberOfEntries - 1 ) {
  #       #Args:             $chr,       $trackIndex,   $pos,  $trackValue,   $mergeFunc, $skipCommit
  #       $self->db->dbPatch($chr, $self->dbName, $pos, $nearestNumber, undef, $count < $self->commitEvery);
  #       $count = $count < $self->commitEvery ? $count + 1 : 0;
  #     }
  #   }
  # }

  # $self->log('info', $self->name . ": finished _writeNearest for $chr");
}

__PACKAGE__->meta->make_immutable;
1;
