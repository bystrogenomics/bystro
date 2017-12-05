use 5.10.0;
use strict;
use warnings;

package Seq::Tracks::Nearest::Build;

our $VERSION = '0.001';

# ABSTRACT: Builds region-based tracks , from some coordinate to some other coordinate
# It then writes a reference to some index in a region database
# And then creates that region database
# The reference # to the region database actually is a unique combination
# of overlapping data based on those coordinates
# So if 3 transcripts overlap completely, there will be a single reference
# that references the combined information of those 3 transcripts (whatever features requested by user)
# Notably, this data is also unique; namely, if the transcript features are all
# redundant, only 1 set of data will be written to the region database at that reference
# If the combinations are not totally redundant, all N unique combinatins will be written
# Finally, within any feature, if all values are completely redundant, only 1 such value
# will be written
# This is distinct from the gene track, which does no de-duplication
# Therefore it makes more sense to store information that is say only available at the gene
# rather than the transcript level, as this kind of track, rather than a type:gene
# It will make that data far more human readable.

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

# Should we store not just the stuff that is intergenic (or between defined regions)
# but also that within the regions themselves
# This is Cutler/Wingo's preferred solution, so that we have data
# for every position in genome
# This I think is reasonable, and from my perspective provides a nice search advantage
# We can search for nearest.dist < 5000 for instance, and include things that are 0 distance away
has storeOverlap => (is => 'ro', isa => 'Bool', default => 1);
has storeNearest => (is => 'ro', isa => 'Bool', default => 1);
my $txNumberKey = 'txNumber';

around BUILDARGS => sub {
  my($orig, $self, $data) = @_;

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
  if($data->{build_field_transformations}{$data->{from}} && (!$data->{to} || $data->{build_field_transformations}{$data->{to}})) {
    return $self->$orig($data);
  }

  # We softly enforce build_field_transformations for the comm
  if($data->{from} eq 'txEnd' || $data->{from} eq 'cdsEnd') {
    $data->{build_field_transformations}{$data->{from}} = '- 1';
  }

  if($data->{to} && ($data->{to} eq 'txEnd' || $data->{to} eq 'cdsEnd')) {
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

  my @fieldDbNames = sort { $a <=> $b } map { $self->getFieldDbName($_) } @{$self->features};

  $pm->run_on_finish( sub {
    my ($pid, $exitCode, $fileName, $exitSignal, $coreDump) = @_;

    if($exitCode != 0) {
      my $err = $self->name . ": got exitCode $exitCode for $fileName: $exitSignal . Dump: $coreDump";

      $self->log('fatal', $err);
    }

    #Only message that is different, in that we don't pass the $fileName
    $self->log('info', $self->name . ": completed building from $fileName");
  });

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

      # If the user wanted to transform the input field names, do, so source field names match
      # those expected by the track
      my @fields = map{ $self->fieldMap->{$_} || $_ } split('\t', $firstLine);

      # Store all features we can find, for Seq::Build::Gene::TX. Avoid autocracy,
      # don't need to know what Gene::TX requires.
      my $fieldIdx = 0;
      for my $field (@fields) {
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

        my @rowData;

        # Field db names are numerical, from 0 to N - 1
        # Assign the last one as the last index, rather than by $#fieldDbNames
        # so that if we have sparse feature names, rowData still can accomodate them
        $#rowData = $fieldDbNames[-1];
        ACCUM_VALUES: for my $fieldName (keys %regionIdx) {
          my $data = $fields[ $regionIdx{$fieldName} ];

          # say split, etc; comes first so that each individual value
          # in an array (if split) can be coerced
          if($self->hasTransform($fieldName) ) {
            $data = $self->transformField($fieldName, $data);
          }

          # convert the value into some type, typically number(N)
          $data = $self->coerceFeatureType($fieldName, $data);

          # if this is a field that we need to store in the region db
          # create a shortened field name
          my $fieldDbName = $self->getFieldDbName($fieldName);

          #store under a shortened fieldName to save space in the db
          $rowData[$fieldDbName] = $data;
        }

        my $from = $rowData[$fromDbName];
        my $to = $rowData[$toDbName];

        if( !(defined $from && defined $to && looks_like_number($from) && looks_like_number($to)) ) {
          $self->log('fatal', "Expected numeric 'from' and 'to' fields, found: $from and $to");
        }

        $regionData{$wantedChr}{$rowIdx} = [$rowIdx, \@rowData];

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
        $self->_writeNearestData($chr, $regionData{$chr}, \@fieldDbNames);

        # We've finished with 1 chromosome, so write that to meta to disk
        $self->completionMeta->recordCompletion($chr);

        $self->db->cleanUp($chr);
      }

      #Commit, sync everything, including completion status, and release mmap
      $self->db->cleanUp();
    $pm->finish(0);
  }

  $pm->wait_all_children;
  return;
}

# We tile in the following way
#---previousLongestEnd##########midpoint#########currentStart-----currentLongestEnd
#everything before midpoint is assigned to the previous region
#everything midpoint on is assigned to the transcripts/regions overlapping currentStart
#everything between currentStart and currentEnd (closed interval)
#is assigned on a base by base basis

# For all transcript/region records that overlap we take the smallest start
# and the largest end, to come up with the largest contiguous region
# And when (in getter) calculating distance, consider anything within
# such an interval as having distance 0

# Similarly, for any transcripts/regions sharing a start with multiple ends
# take the largest end
# And for any ends sharing a start, take the smallest start
sub _writeNearestData {
  my ($self, $chr, $regionDataHref, $fieldDbNames) = @_;

  my $fromDbName = $self->getFieldDbName($self->from);
  my $toDbName = $self->getFieldDbName($self->to);

  my $regionDbName = $self->regionTrackPath($chr);

  my $uniqNumMaker = _getTxNumber();
  my $uniqRegionEntryMaker = _makeUniqueRegionData($fromDbName, $toDbName, $fieldDbNames);

  # First sort by to position, ascending (smallest to largest)
  # then by from position (smallest to largest)
  my @sorted = sort { $a->[1][$fromDbName] <=> $b->[1][$fromDbName] } sort { $a->[1][$toDbName] <=> $b->[1][$toDbName] } values %{$regionDataHref};

  # the tx starts ($self->from key)
  my %startData;

  for my $data (@sorted) {
    my $start = $data->[1][$fromDbName];
    push @{$startData{$start}}, $data;
  }

  # the tx ends ($self->to key)
  my %endData;

  for my $data (@sorted) {
    my $end = $data->[1][$toDbName];
    push @{$endData{$end}}, $data;
  }

  $self->log('info', $self->name . ": starting for $chr");

  # Get database length : assumes reference track already in the db
  my $genomeNumberOfEntries = $self->db->dbGetNumberOfEntries($chr);

  if(!$genomeNumberOfEntries) {
    $self->log('fatal', $self->name . " requires at least the reference track, to know how many bases in $chr");
  }

  # Track the longest (further in db toward end of genome) txEnd, because
  #  in  case of overlapping transcripts, want the points that ARENT 
  #  covered by a gene (since those have apriori nearest records: themselves)
  #  This also acts as our starting position
  # my $longestPreviousTxEnd = 0;
  # my $longestPreviousTxNumber;

  my $midPoint;

  # We will combine overlapping transcripts here, and generate a unique txNumber
  # for each combination
  # This saves us having to walk transcript arrays to gather features at run time
  # And also saves us having to write arrays to the main database, per genome position
  # A big net win
  my @globalTxData;

  my $previousLongestEnd;
  my $previousTxNumber;
  # track what was previously made
  my %completed;
  TXSTART_LOOP: for (my $n = 0; $n < @sorted; $n++) {
    my $start = $sorted[$n][1][$fromDbName];

    # if(ref $startData{$start}) {
    #   $longestEnd = max ( map { $_->[1][$toDbName]} ) @{$startData{$start}};
    # }

    # If > 1 transcript shares a start, txNumber will be an array of numbers
    # <ArrayRef[Int]> of length 1 or more
    # else will be a scalar, save some space in db, and reduce Perl memory growth

    # Assign a unique txNumber based on the overlap of transcripts
    # Idempotent
    my @stuff = map { $_->[0] } @{$startData{$start}};

    my $txNumber = $uniqNumMaker->([ map { $_->[0] } @{$startData{$start}} ]);

    # If we're 1 short of the new txNumber (index), we have some unique data
    # add the new item
    if(@globalTxData == $txNumber) {
      my $combinedValues = $uniqRegionEntryMaker->([ map { $_->[1] } @{$startData{$start}} ]);
      # p $combinedValues;
      # Since these values don't nec share an end, take the max one for the overlap
      $combinedValues->[$toDbName] = ref $combinedValues->[$toDbName] ? max(@{$combinedValues->[$toDbName]}) : $combinedValues->[$toDbName];

      # write the region database, storing the region data at our sequential txNumber, allow us to release the data
      $self->db->dbPut($regionDbName, $txNumber, $combinedValues);

      # we only need to store the longest end; only value that is needed below from combinedValues
      push @globalTxData, $combinedValues->[$toDbName];
    }

    if(defined $previousLongestEnd) {
      # Here we can assume that both $start and $longestPreviousEnd are both 0-based, closed
      # and so the first intergenic base is + 1 of the longestPreviousTxEnd and - 1 of the current start
      #say previousLongestEnd == 1
      #say $start == 11
      #end..2..3..4..5..Midpoint..7..8..9..10..start
      #(11 - 1 ) / 2 == 5; 5 + end = 5 + 1 == 6 == midpoint
      # unnecessary extra precedence parenthesis, makes me feel safer :|
      $midPoint = $previousLongestEnd + ( ($start - $previousLongestEnd) / 2 );
    }

    #### Accumulate txNumber or longestPreviousTxNumber for positions between transcripts #### 

    # If we have no previous end or midpoint, we're starting from 0 index in db
    # and moving until the $start
    $previousLongestEnd //= -1;
    $midPoint //= -1;

    # Consider/store intergenic things (note: if previousLongestEnd > $start, last tx overlapped this one)
    if($self->storeNearest && $previousLongestEnd < $start) {
      # we force both the end and start to be 0-based closed, so start from +1 of previous end
      # and - 1 of the start
      POS_LOOP: for my $pos ( $previousLongestEnd + 1 .. $start - 1 ) {
        if($pos >= $midPoint) {
          #Args:             $chr,       $trackIndex,   $pos,  $trackValue, $mergeFunc, $skipCommit
          $self->db->dbPatch($chr, $self->dbName, $pos, $txNumber);
        } else {
          #Args:             $chr,       $trackIndex,   $pos,  $trackValue,             $mergeFunc, $skipCommit
          $self->db->dbPatch($chr, $self->dbName, $pos, $previousTxNumber);
        }
      }
    }

    my $longestEnd = $globalTxData[$txNumber];

    # If we want to store the stuff in the regions themselves, do that
    if($self->storeOverlap) {
      # We investigate everything from the present tx down;
      my @data = @sorted[$n .. $#sorted];

      # Remember, here longest end is 0-based, closed (last pos is the last 0-based
      # position in the transcript)
      for my $pos ($start .. $longestEnd) {
        # There may be overlaps between adjacent groups of transcripts
        # Since we search for all overlapping transcripts for every position
        # once we've visited one position, we need never visit it again
        if($completed{$pos}) {
          next;
        }

        my @overlap;
        
        I_LOOP: for my $iRow (@data) {
          my $iFrom = $iRow->[1][$fromDbName];
          my $iTo = $iRow->[1][$toDbName];

          if($pos >= $iFrom && $pos <= $iTo) {
            push @overlap, $iRow;
          } elsif($iFrom > $pos) {
            last I_LOOP;
          }
        }

        if(!@overlap) {
          say "no overlap from $start to $longestEnd";
        }

        # Make a unique overlap combination
        my $txNumber = $uniqNumMaker->([map { $_->[0] } @overlap]);

        # If we're 1 short of the new txNumber (index), we have some unique data
        # add the new item
        if(@globalTxData == $txNumber) {
          my $combinedValues = $uniqRegionEntryMaker->([map { $_->[1] } @overlap]);

          # Since these values don't nec share either start or end, take the max of each for the overlap
          # Therefore, when checking dist, we'll be able to check scalars
          $combinedValues->[$fromDbName] = ref $combinedValues->[$fromDbName] ? min(@{$combinedValues->[$fromDbName]}) : $combinedValues->[$fromDbName];
          $combinedValues->[$toDbName] = ref $combinedValues->[$toDbName] ? max(@{$combinedValues->[$toDbName]}) : $combinedValues->[$toDbName];

          # write the region database, storing the region data at our sequential txNumber, allow us to release the data
          $self->db->dbPut($regionDbName, $txNumber, $combinedValues);

          # we only need to store the longest end; only value that is needed from combinedValues
          push @globalTxData, $combinedValues->[$toDbName];
        }

        # Assign the transcript number
        $self->db->dbPatch($chr, $self->dbName, $pos, $txNumber);

        $completed{$pos} = 1;
      }
    }

    ###### Store the previous values for the next loop's midpoint calc ######
    my $longestEndTxNumber = $uniqNumMaker->([ map { $_->[0] } @{$endData{$longestEnd}} ]);

    if(@globalTxData == $longestEndTxNumber) {
      my $combinedValues = $uniqRegionEntryMaker->([ map { $_->[1] } @{$endData{$longestEnd}} ]);

      # Since these don't nec. share a start, choose the min start for the overlap
      $combinedValues->[$fromDbName] = ref $combinedValues->[$fromDbName] ? min(@{$combinedValues->[$fromDbName]}) : $combinedValues->[$fromDbName];

      # write the region database, storing the region data at our sequential txNumber, allow us to release the data
      $self->db->dbPut($regionDbName, $longestEndTxNumber, $combinedValues);

      # we only need to store the longest end; only value that is needed from combinedValues
      # TODO: Should this be $longestEnd?
      push @globalTxData, $combinedValues->[$toDbName];
    }

    $previousTxNumber = $longestEndTxNumber;
    $previousLongestEnd = $longestEnd;
  }

  if($self->storeNearest) {
    # Once we've reached the last transcript, we still likely have some data remaining
    END_LOOP: for my $pos ( $previousLongestEnd + 1 .. $genomeNumberOfEntries - 1 ) {
      #Args:             $chr,       $trackIndex,   $pos,  $trackValue,   $mergeFunc, $skipCommit
      $self->db->dbPatch($chr, $self->dbName, $pos, $previousTxNumber);
    }
  }

  $self->log('info', $self->name . ": finished for $chr");
}

sub _getTxNumber {
  my $txNumber = 0;
  my %uniqCombos;

  return sub {
    my $numAref = shift;

    my $hash = md5(@$numAref);

    if(defined $uniqCombos{$hash}) {
      return $uniqCombos{$hash};
    }

    $uniqCombos{$hash} = $txNumber;

    $txNumber++;

    return $uniqCombos{$hash};
  }
}

sub _makeUniqueRegionData {
  my ($fromDbName, $toDbName, $featureKeysAref) = @_;

  my @featureKeys = @$featureKeysAref;
  return sub {
    my $aRef = shift;

    my %dup;

    my @out;

    # Assumes arrays of equal length
    #Expects val to have:
    for my $val (@$aRef) {
      my %uniqueRows;

      # Figure out what is unique by building an array that does not include
      # the to and from positoins
      # Since the md5 functon will complain about sparse arrays
      # fill missing values with "" during the md5 check
      # However, in the final, unique output, undefined values will remain undefined
      my @nonFromTo;
      
      for my $i (@featureKeys) {
        if($i != $fromDbName && $i != $toDbName) {
          push @nonFromTo, $val->[$i] || "";
        }
      }

      my $hash = md5(@nonFromTo);

      if($dup{$hash}) {
        next;
      }

      $dup{$hash} = 1;

      for my $intKey (@featureKeys) {
        push @{$out[$intKey]}, $val->[$intKey];
      }
    }

    for my $intKey (@featureKeys) {
      if(!ref $out[$intKey]) {
        next;
      }

      if(@{$out[$intKey]} == 1) {
        $out[$intKey] = $out[$intKey][0];
        next;
      }

      my @uniq = uniq(@{$out[$intKey]});

      # Don't lose the relationships between data
      # If there's a single unique value, use that however to remove redundancy
      if(@uniq == 1) {
        $out[$intKey] = $uniq[0];
      }
    }

    return \@out;
  }
}
__PACKAGE__->meta->make_immutable;
1;
