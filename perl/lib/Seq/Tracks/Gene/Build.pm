use 5.10.0;
use strict;
use warnings;

package Seq::Tracks::Gene::Build;

our $VERSION = '0.001';

# ABSTRACT: Builds Gene Tracks
# Stores refSeq data, and generates in-silico transcribed transcripts (and associated fields)

use Mouse 2;
use namespace::autoclean;

use Parallel::ForkManager;

use Seq::Tracks::Gene::Build::TX;
use Seq::Tracks::Gene::Definition;
use Seq::Tracks;

extends 'Seq::Tracks::Build';
#exports regionTrackPath
with 'Seq::Tracks::Region::RegionTrackPath';

use List::Util qw/first/;

my $geneDef = Seq::Tracks::Gene::Definition->new();

# We don't remap field names
# It's easier to remember the real names than real names + our domain-specific names

# Required, or must be mapped using 'fieldMap' in YAML
my @coordinateFields = (
  'chrom',  'txStart',    'txEnd',    'cdsStart',
  'cdsEnd', 'exonStarts', 'exonEnds', 'strand'
);

my %coordinateFields = map { $_ => 1 } @coordinateFields;

my %coordinateTransforms = (
  exonStarts => sub {
    return [ split ',', $_[0] ];
  },
  exonEnds => sub {
    return [ split ',', $_[0] ];
  }
);

# has chrom_field_name => (is => 'ro', lazy => 1, default => 'chrom' );
# has txStart_field_name => (is => 'ro', lazy => 1, default => 'txStart' );
# has txEnd_field_name => (is => 'ro', lazy => 1, default => 'txEnd' );
# has cdsStart_field_name => (is => 'ro', lazy => 1, default => 'txStart' );
# has cdsEnd_field_name => (is => 'ro', lazy => 1, default => 'txEnd' );
# has exonStarts_field_name => (is => 'ro', lazy => 1, default => 'txStart' );
# has txEnd_field_name => (is => 'ro', lazy => 1, default => 'txEnd' );

has build_region_track_only =>
  ( is => 'ro', isa => 'Bool', coerce => 1, lazy => 1, default => 0 );
has join => ( is => 'ro', isa => 'HashRef' );

# These are the features stored in the Gene track's region database
# Does not include $geneDef->txErrorName here, because that is something
# that is not actually present in UCSC refSeq or knownGene records, we add ourselves
has '+features' => ( default => sub { $geneDef->ucscGeneAref; } );

my $txNumberKey = 'txNumber';
my $joinTrack;

sub BUILD {
  my $self = shift;

  $self->getFieldDbName( $geneDef->txErrorName );
  # Do this before we build, to avoid threading-related issues
  for my $f (@coordinateFields) {
    $self->getFieldDbName($f);
  }
}

# 1) Store a reference to the corresponding entry in the gene database (region database)
# 2) Store this codon information at some key, which the Tracks::Region::Gene
# 3) Store transcript errors, if any
# 4) Write region data
# 5) Write gene track data in main db
# 6) Write nearest genes if user wants those
sub buildTrack {
  my $self = shift;
  # txErrorName isn't a default feature, initializing here to make sure
  # we store this value (if calling for first time) before any threads get to it
  my @allFiles = $self->allLocalFiles;

  # Only allow 1 thread because perl eats memory like candy
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

  # Assume one file per loop, or all sites in one file. Tracks::Build warns if not
  for my $file (@allFiles) {
    # Although this should be unnecessary, environments must be created
    # within the process that uses them
    # This provides a measure of safety
    $self->db->cleanUp();

    $pm->start($file) and next;
    my ( $seenChrs, $data );

    my ( $err, undef, $fh ) = $self->getReadFh($file);

    if ($err) {
      $self->log( 'fatal', $self->name . ": $err" );
    }

    ( $err, $seenChrs, $data ) = $self->_readTxData($fh);

    if ($err) {
      $self->log( 'fatal', $self->name . ": $err" );
    }

    # If we fork a process in order to read (example zcat) prevent that process
    # from becoming defunct
    $self->safeCloseBuilderFh( $fh, $file, 'fatal' );

    my %all      = %{ $data->{all} };
    my %txStarts = %{ $data->{txStart} };
    my %regions  = %{ $data->{region} };

    # If we skipped the $file, will never get here, so this is an error
    # Can happen if only 1 file, with only 1 chromosome (not building all chrs)
    if ( !%all && $seenChrs == 1 ) {
      # This returns to parent, in $pm->run_on_finish
      $pm->finish(0);
    }

    if ( !%all ) {
      $self->log( 'fatal', $self->name . ": no transcript data accumulated" );
    }

    ############################### Make transcripts #########################
    my %visitedChrs;

    my @allChrs = keys %all;

    my $txErrorDbname = $self->getFieldDbName( $geneDef->txErrorName );

    TX_LOOP: for my $chr (@allChrs) {
      $visitedChrs{$chr} //= 1;
      # We may want to just update the region track,
      # TODO: Note that this won't store any txErrors
      if ( $self->build_region_track_only ) {
        $self->_writeRegionData( $chr, $regions{$chr} );

        if ( $self->join ) {
          $self->_joinTracksToGeneTrackRegionDb( $chr, $txStarts{$chr} );
        }

        next TX_LOOP;
      }

      $self->log( 'info', $self->name . ": starting to build transcripts for $chr" );

      my @allTxStartsAscending = sort { $a <=> $b } keys %{ $all{$chr} };

      for my $txStart (@allTxStartsAscending) {
        for my $txData ( @{ $all{$chr}->{$txStart} } ) {
          my $txNumber = $txData->{$txNumberKey};

          # Important that anything this class needs to commit to db happens
          # before we open our cursor
          my $txInfo = Seq::Tracks::Gene::Build::TX->new($txData);

          if ( @{ $txInfo->transcriptSites } % 2 != 0 ) {
            $err = $self->name
              . ": expected txSiteDataAndPos to contain (position1, value1, position2, value2) data";
            $self->log( 'fatal', $err );
          }

          # Note that is Build::TX doesn't close it txn, this will be
          # impossible; may be an LMDB_File bug
          # If that is problematic, placing this before Build::TX will work
          # for some reason, order of SubTxn's matters
          # https://github.com/salortiz/LMDB_File/issues/30
          my $cursor = $self->db->dbStartCursorTxn($chr);

          INNER: for ( my $i = 0; $i < @{ $txInfo->transcriptSites }; $i += 2 ) {
            # $txInfo->transcriptSites->[$i] corresponds to $pos, $i + 1 to value
            # Commit for every position
            # This also ensures that $mainMergeFunc will always be called with fresh data
            $self->db->dbPatchCursorUnsafe(
              $cursor,
              $chr,
              $self->dbName,
              #pos
              $txInfo->transcriptSites->[$i],
              #value <Array[Scalar]>
              $txInfo->transcriptSites->[ $i + 1 ],
              #how we handle cases where multiple overlap
              \&_dbMergeFunc
            );
          }

          # Commits, closes cursor every transcript
          $self->db->dbEndCursorTxn($chr);

          if ( @{ $txInfo->transcriptErrors } ) {
            my @errors = @{ $txInfo->transcriptErrors };
            $regions{$chr}->{$txNumber}{$txErrorDbname} = \@errors;
          }
        }

        delete $all{$chr}->{$txStart};
      }

      delete $all{$chr};

      $self->_writeRegionData( $chr, $regions{$chr} );

      delete $regions{$chr};

      if ( $self->join ) {
        $self->_joinTracksToGeneTrackRegionDb( $chr, $txStarts{$chr} );
      }

      delete $txStarts{$chr};

      $self->log( 'info',
        $self->name . ": finished building transcripts for $chr from $file" );

      $self->db->cleanUp();
    }

    #Commit, sync everything, including completion status, and release mmap
    $self->db->cleanUp();

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

sub _dbMergeFunc {
  # Only when called when there is a defined $oldVal
  my ( $chr, $pos, $oldVal, $newVal ) = @_;
  # make it an array of arrays (array of geneTrack site records)
  if ( !ref $oldVal->[0] ) {
    return ( undef, [ $oldVal, $newVal ] );
  }

  #oldVal is an array of arrays, push on to it
  my @updatedVal = @$oldVal;

  push @updatedVal, $newVal;

  #TODO: Should we throw any errors?
  return ( undef, \@updatedVal );
}

sub _getIdx {
  my ( $self, $firstLine ) = @_;

  chomp $firstLine;

  my ( %allIdx, %regionIdx );
  # If the user wanted to transform the input field names, do, so source field names match
  # those expected by the track
  my @fields = map { $self->fieldMap->{$_} || $_ } split( '\t', $firstLine );

  # Store all features we can find, for Seq::Build::Gene::TX. Avoid autocracy,
  # don't need to know what Gene::TX requires.
  my $fieldIdx = 0;
  for my $field (@fields) {
    $allIdx{$field} = $fieldIdx;
    $fieldIdx++;
  }

  my $err;

  # Some featuers are core to transcript building
  for my $reqField (@coordinateFields) {
    if ( !defined $allIdx{$reqField} ) {
      $err = 'Must provide, or map via "fieldMap" in the YAML config
        the following fields: ' . join( ',', @coordinateFields );

      return ( $err, undef );
    }
  }

  # Region database features; as defined by user in the YAML config, or our default
  REGION_FEATS: for my $field ( @{ $self->features } ) {
    if ( exists $allIdx{$field} ) {
      $regionIdx{$field} = $allIdx{$field};
      next REGION_FEATS;
    }

    $err = $self->name . ": required $field missing in header: $firstLine";
    last;
  }

  return (
    $err,
    {
      all    => \%allIdx,
      region => \%regionIdx,
    }
  );
}

sub _readTxData {
  my ( $self, $fh ) = @_;

  my $firstLine = <$fh>;

  # Fatal/exit will only affect that process, won't affect others
  if ( !defined $firstLine ) {
    return "Couldn't read header";
  }

  # support non-Unix line endings
  my $err = $self->setLineEndings($firstLine);

  if ($err) {
    return $err;
  }

  ( $err, my $idx ) = $self->_getIdx($firstLine);

  my %allIdx    = %{ $idx->{all} };
  my %regionIdx = %{ $idx->{region} };

  if ($err) {
    return $err;
  }

  my ( $wantedChr, $txNumber, $txStart, $txEnd, $chr, @fields );

  my ( %seenChrsInFile, %txNumbers, %regionData, %txStartData, %allData );

  FH_LOOP: while (<$fh>) {
    chomp;
    @fields = split( '\t', $_ );

    $chr     = $fields[ $allIdx{'chrom'} ];
    $txStart = $fields[ $allIdx{'chrom'} ];

    $seenChrsInFile{$chr} //= 1;

    # Normalize the representation, such that having or missing 'chr'
    # or using MT instead of M won't matter
    $chr = $self->normalizedWantedChr->{$chr};

    # chr may be undefined
    if ( !defined $wantedChr || ( !defined $chr || $wantedChr ne $chr ) ) {
      $wantedChr = $self->chrWantedAndIncomplete($chr);
    }

    if ( !defined $wantedChr ) {
      next FH_LOOP;
    }

    # Keep track of our 0-indexed transcript reference numbers
    if ( !$txNumbers{$wantedChr} ) {
      $txNumbers{$wantedChr} = 0;
    }

    $txNumber = $txNumbers{$wantedChr};

    my %rowData;

    my $fieldDbName;
    ACCUM_VALUES: for my $fieldName ( keys %allIdx ) {
      if ( $self->hasTransform($fieldName) ) {
        $fields[ $allIdx{$fieldName} ] =
          $self->transformField( $fieldName, $fields[ $allIdx{$fieldName} ] );
      }

      # TODO: This could lead to surprising behavior
      # User can modify this behavior by passing a transform above
      # But the expectation of $coordinateTransforms{$fieldName} should be
      # documented, since not all transforms will work
      if ( $coordinateTransforms{$fieldName} ) {
        $fields[ $allIdx{$fieldName} ] =
          $coordinateTransforms{$fieldName}->( $fields[ $allIdx{$fieldName} ] );
      }

      my $data = $self->coerceFeatureType( $fieldName, $fields[ $allIdx{$fieldName} ] );

      if ( !defined $data ) {
        next ACCUM_VALUES;
      }

      # if this is a field that we need to store in the region db
      # create a shortened field name
      $fieldDbName = $self->getFieldDbName($fieldName);

      $rowData{$fieldName} = $data;

      if ( !( defined $regionIdx{$fieldName} || defined $coordinateFields{$fieldName} ) ) {
        next ACCUM_VALUES;
      }

      #store under a shortened fieldName to save space in the db
      $regionData{$wantedChr}->{$txNumber}{$fieldDbName} = $data;
    }

    $txStart = $rowData{'txStart'};

    if ( !defined $txStart ) {
      return ': missing transcript start ( we expected a value @ ' . 'txStart' . ')';
    }

    $txEnd = $rowData{'txEnd'};

    if ( !defined $txEnd ) {
      return 'missing transcript start ( we expected a value @ ' . 'txEnd' . ')';
    }

    #a field added by Bystro
    # $regionData{$wantedChr}->{$txNumber}{$self->getFieldDbName($geneDef->txSizeName)} = $txEnd + 1 - $txStart;

    if ( defined $txStartData{$wantedChr}{$txStart} ) {
      push @{ $txStartData{$wantedChr}{$txStart} }, [ $txNumber, $txEnd ];
    }
    else {
      $txStartData{$wantedChr}{$txStart} = [ [ $txNumber, $txEnd ] ];
    }

    $rowData{$txNumberKey} = $txNumber;

    push @{ $allData{$wantedChr}{$txStart} }, \%rowData;

    $txNumbers{$wantedChr} += 1;
  }

  my $seenChrs = keys %seenChrsInFile;

  my %out;

  $out{txStart} = \%txStartData;
  $out{all}     = \%allData;
  $out{region}  = \%regionData;

  return ( $err, $seenChrs, \%out );
}

sub _writeRegionData {
  my ( $self, $chr, $regionsHref ) = @_;

  $self->log( 'info', $self->name . ": starting _writeRegionData for $chr" );

  my $dbName = $self->regionTrackPath($chr);

  my @txNums = sort { $a <=> $b } keys %$regionsHref;

  for my $txNumber (@txNums) {
    # Patch one at a time, because we assume performance isn't an issue
    # And neither is size, so hash keys are fine
    # TODO: move away from this; don't store any hashes, use arrays
    $self->db->dbPatchHash( $dbName, $txNumber, $regionsHref->{$txNumber} );
  }

  $self->log( 'info', $self->name . ": finished _writeRegionData for $chr" );
  return;
}

############ Joining some other track to Gene track's region db ################

# TODO: Add check to see if values have already been entered
sub _joinTrackMergeFunc {
  my ( $chr, $pos, $oldVal, $newVal ) = @_;

  my @updated;

  #If the old value is an array, push the new values on to the old values
  if ( ref $oldVal ) {
    @updated = @$oldVal;

    for my $val ( ref $newVal ? @$newVal : $newVal ) {
      if ( !defined $val ) {
        next;
      }

      push @updated, $val;
    }
  }
  else {
    if ( defined $oldVal ) {
      @updated = ($oldVal);
    }

    for my $val ( ref $newVal ? @$newVal : $newVal ) {
      if ( !defined $val ) {
        next;
      }

      # If not array I want to see an error
      push @updated, $val;
    }
  }

  # Try to add as little junk as possible
  if ( @updated == 0 ) {
    return ( undef, $oldVal );
  }

  if ( @updated == 1 ) {
    return ( undef, $updated[0] );
  }

  return ( undef, \@updated );
}

sub _joinTracksToGeneTrackRegionDb {
  my ( $self, $chr, $txStarts ) = @_;

  if ( !$self->join ) {
    return $self->name . ": join not set in _joinTracksToGeneTrackRegionDb";
  }

  my $tracks = Seq::Tracks->new();
  $joinTrack = $tracks->getTrackBuilderByName( $self->joinTrackName );

  if ( !$joinTrack ) {
    return
        $self->name
      . ': join track '
      . $self->joinTrackName
      . ' has no "tracks" .yml entry';
  }

  $self->log( 'info',
        $self->name
      . ": starting _joinTracksToGeneTrackRegionDb "
      . "for $chr using: "
      . $self->joinTrackName );

  # Gene tracks cover certain positions, record the start and stop
  my @positionRanges;
  my @txNums;

  for my $txStart ( keys %$txStarts ) {
    foreach ( @{ $txStarts->{$txStart} } ) {
      my $txNumber = $_->[0];
      my $txEnd    = $_->[1];
      push @positionRanges, [ $txStart, $txEnd ];
      push @txNums,         $txNumber;
    }
  }

  my $dbName = $self->regionTrackPath($chr);

  # For each txNumber, run dbPatchHash on any joining data
  $joinTrack->joinTrack(
    $chr,
    \@positionRanges,
    $self->joinTrackFeatures,
    sub {
      # Called every time a match is found
      # Index is the index of @ranges that this update belongs to
      my ( $hrefToAdd, $index ) = @_;

      my %out;
      foreach ( keys %$hrefToAdd ) {
        if ( !defined $hrefToAdd->{$_} ) {
          next;
        }

        if ( ref $hrefToAdd->{$_} eq 'ARRAY' ) {
          my @arr;
          my %uniq;
          for my $entry ( @{ $hrefToAdd->{$_} } ) {
            if ( !defined $entry ) {
              next;
            }

            if ( $uniq{$entry} ) {
              next;
            }

            push @arr, $entry;
            $uniq{$entry} = 1;
          }

          # Don't add empty arrays to the database
          $hrefToAdd->{$_} = @arr ? \@arr : undef;
        }

        if ( defined $hrefToAdd->{$_} ) {
          # Our LMDB writer requires a value, so only add to our list of db entries
          # to update if we have a value
          #$self->getFieldDbName generates a name for the field we're joining, named $_
          $self->db->dbPatchHash( $dbName, $txNums[$index],
            { $self->getFieldDbName($_) => $hrefToAdd->{$_} },
            \&_joinTrackMergeFunc );
        }
      }

      # Free memory as soon as possible
      undef $hrefToAdd;
      undef %out;
    }
  );

  $self->log( 'info',
    $self->name . ": finished _joinTracksToGeneTrackRegionDb for $chr" );
  return;
}

__PACKAGE__->meta->make_immutable;
1;
