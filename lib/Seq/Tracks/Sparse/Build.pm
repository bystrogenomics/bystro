use 5.10.0;
use strict;
use warnings;
package Seq::Tracks::Sparse::Build;

our $VERSION = '0.001';

=head1 DESCRIPTION

  @class Seq::Tracks::SparseTrack::Build
  Builds any sparse track

=cut

# TODO: better error handling. Check for # completed, rather than # failed
# in some cases, errors thrown don't actually get caught with exitCode other than 0
use Mouse 2;

use namespace::autoclean;
use List::MoreUtils qw/firstidx/;
use Parallel::ForkManager;
use Scalar::Util qw/looks_like_number/;

use DDP;

extends 'Seq::Tracks::Build';

# We assume sparse tracks have at least one feature; can remove this requirement
# But will need to update makeMergeFunc to not assume an array of values (at least one key => value)
has '+features' => (required => 1);

#These cannot be overriden (change in api), but users can use fieldMap to rename any field
#in the input file
has chromField => (is => 'ro', isa => 'Str', init_arg => undef, lazy => 1, default => 'chrom');
has chromStartField => (is => 'ro', isa => 'Str', init_arg => undef, lazy => 1, default => 'chromStart');
has chromEndField => (is => 'ro', isa => 'Str', init_arg => undef, lazy => 1, default => 'chromEnd');

# We skip entries that span more than this number of bases
has maxVariantSize => (is => 'ro', isa => 'Int', lazy => 1, default => 32);

################# Private ################
# Only 0 based files should be half closed
has _halfClosedOffset => (is => 'ro', init_arg => undef, writer => '_setHalfClosedOffset');

sub BUILD {
  my $self = shift;

  $self->_setHalfClosedOffset($self->based == 0 ? 1 : 0);

  if($self->based != 1 && $self->based != 0) {
    $self->log('fatal', $self->name . ": SparseTracks expect based to be 0 or 1");
    die $self->name . ": SparseTracks expect based to be 0 or 1";
  }
}

sub buildTrack {
  my $self = shift;

  my $pm = Parallel::ForkManager->new($self->max_threads);

  # Get an instance of the merge function that closes over $self
  # Note that tracking which positinos have been over-written will only work
  # if there is one chromosome per file, or if all chromosomes are in one file
  # At least until we share $madeIntoArray (in makeMergeFunc) between threads
  # Won't be an issue in Go
  my ($mergeFunc, $cleanUpMerge) = $self->makeMergeFunc();

  my %completedDetails;
  $pm->run_on_finish(sub {
    my ($pid, $exitCode, $fileName, undef, undef, $errOrChrs) = @_;

    if($exitCode != 0) {
      my $err = $errOrChrs ? "due to: $$errOrChrs" : "due to an untimely demise";

      $self->log('fatal', $self->name . ": Failed to build $fileName $err");
      die $self->name . ": Failed to build $fileName $err";
    }

    for my $chr (keys %$errOrChrs) {
      if(!$completedDetails{$chr}) {
        $completedDetails{$chr} = [$fileName];
      } else {
        push @{$completedDetails{$chr}}, $fileName;
      }
    }

    $self->log('info', $self->name . ": completed building from $fileName");
  });

  for my $file (@{$self->local_files}) {
    $self->log('info', $self->name . ": beginning building from $file");

    $pm->start($file) and next;
      my $fh = $self->get_read_fh($file);

      ############# Get Headers ##############
      my $firstLine = <$fh>;

      # Support non-unix line endings
      my $err = $self->setLineEndings($firstLine);

      if($err) {
        $self->log('fatal', $err);
      }

      if(!$firstLine) {
        my $err;
        if(!close($fh) && $? != 13) {
          $err = $self->name . ": failed to open $file due to $1 ($?)";
        } else {
          $err = $self->name . ": $file empty";
        }

        $self->log('fatal', $err);
      }

      my ($featureIdxHref, $reqIdxHref, $fieldsToTransformIdx, $fieldsToFilterOnIdx, $numColumns) = 
        $self->_getHeaderFields($file, $firstLine, $self->features);
      ############## Read file and insert data into main database #############
      my $wantedChr;

      # Record which chromosomes were recorded for completionMeta
      my %visitedChrs;

      my %fieldDbNames;

      my ($invalid, $failedFilters, $tooLong) = (0, 0, 0);

      my ($chr, @fields, @sparseData, $start, $end);
      FH_LOOP: while ( my $line = $fh->getline() ) {
        chomp $line;

        @fields = split('\t', $line);

        if(! $self->_validLine(\@fields, $., $reqIdxHref, $numColumns) ) {
          $invalid++;
          next FH_LOOP;
        }

        if(! $self->_passesFilter($fieldsToFilterOnIdx, \@fields, $.)) {
          $failedFilters++;
          next FH_LOOP;
        }

        $self->_transform($fieldsToTransformIdx, \@fields);

        # Normalizes the $chr representation to one we may want but did not specify
        # Example: 1 becomes chr1, and is checked against our list of wanted chromosomes
        # Avoids having to use a field transformation, since this may be very common
        # and Bystro typical use is with UCSC-style chromosomes
        # If the chromosome isn't wanted, $chr will be undefined
        $chr = $self->normalizedWantedChr( $fields[ $reqIdxHref->{$self->chromField} ] );

        #If the chromosome is new, write any data we have & see if we want new one
        if( !$wantedChr || ($wantedChr && (!$chr || $wantedChr ne $chr)) ) {
          if($wantedChr) {
            #Commit, flush anything remaining to disk, release mapped memory
            $self->db->cleanUp($wantedChr);
          }

          $wantedChr = $chr && $self->completionMeta->okToBuild($chr) ? $chr : undef;
        }

        if(!$wantedChr) {
          if($self->chrPerFile) {
            $self->log('info', $self->name . ": skipping $file because found unwanted chr, and expect 1 chr per file");

            last FH_LOOP;
          }

          next FH_LOOP;
        }

        ($start, $end) = $self->_getPositions(\@fields, $reqIdxHref);

        if($end + 1 - $start > $self->maxVariantSize) {
          # TODO: think about adding this back in; results in far too many log messages
          # $self->log('debug', "Line spans > " . $self->maxVariantSize . " skipping: $line");
          $tooLong++;
          next FH_LOOP;
        }

        # Collect all of the feature data as an array
        # Coerce the field into the type specified for $name, if coercion exists
        # Perl arrays auto-grow https://www.safaribooksonline.com/library/view/perl-cookbook/1565922433/ch04s04.html
        @sparseData = ();

        # Get the field values after transforming them to desired types
        FNAMES_LOOP: for my $name (keys %$featureIdxHref) {
          my $value = $self->coerceFeatureType( $name, $fields[ $featureIdxHref->{$name} ] );

          if(!exists $fieldDbNames{$name}) {
            $fieldDbNames{$name} = $self->getFieldDbName($name);
          }

          # getFieldDbName will croak if it can't make or find a dbName
          $sparseData[ $fieldDbNames{$name} ] = $value;
        }

        for my $pos (($start .. $end)) {
          # Write every position to disk
          # Commit for each position, fast if using MDB_NOSYNC
          $self->db->dbPatch($wantedChr, $self->dbName, $pos, \@sparseData, $mergeFunc);
        }

        undef @sparseData;
        # Track affected chromosomes for completion recording
        $visitedChrs{$wantedChr} //= 1;
      }

      #Commit, sync everything, including completion status, and release mmap
      $self->db->cleanUp();

      if(!close($fh) && $? != 13) {
        $self->log('fatal', $self->name . ": couldn't close or read $file due to $! ($?)");
      } else {
        $self->log('info', $self->name . ": $file closed with $?");
        $self->log('info', $self->name . ": invalid lines found in $file: $invalid");
        $self->log('info', $self->name . ": lines that didn't pass filters in $file: $failedFilters");
        $self->log('info', $self->name . ": lines that were longer than " . $self->maxVariantSize . " found in $file: $tooLong");
      }

    $pm->finish(0, \%visitedChrs);
  }

  $pm->wait_all_children;

  # Defer recording completion state until all requested files visited, to ensure
  # that if chromosomes are mis-sorted, we still build all that is needed
  for my $chr (keys %completedDetails) {
    $self->completionMeta->recordCompletion($chr);
    $cleanUpMerge->($chr);

    $self->log('info', $self->name . ": recorded $chr completed, from " . (join(",", @{$completedDetails{$chr}})));
  }

  return;
}

# Unlike buildTrack, joinTrack does not use a length filter; huge CNVs will
# be stored
# @param <ArrayRef> $wantedPositionsAref : expects all wanted positions
sub joinTrack {
  my ($self, $wantedChr, $wantedPositionsAref, $wantedFeaturesAref, $callback) = @_;

  if(!$self->chrIsWanted($wantedChr)) {
    $self->log('fatal', $self->name . " join track: called with $wantedChr which is not in our config list of chromosomes");
    die $self->name . " join track: called with $wantedChr which is not in our config list of chromosomes";
  }

  $self->log('info', $self->name . " join track: called for $wantedChr");

  for my $file ($self->allLocalFiles) {
    my $fh = $self->get_read_fh($file);

    ############# Get Headers ##############
    my $firstLine = <$fh>;

    if(!$firstLine) {
      my $err;
      if(!close($fh) && $? != 13) {
        $err = $self->name . " join track: failed to open $file due to $1 ($?)";
      } else {
        $err = $self->name . " join track: $file empty";
      }

      $self->log('fatal', $err);
      die $err;
    }

    my ($featureIdxHref, $reqIdxHref, $fieldsToTransformIdx, $fieldsToFilterOnIdx, $numColumns) = 
      $self->_getHeaderFields($file, $firstLine, $wantedFeaturesAref);

    my @allWantedFeatureIdx = keys %$featureIdxHref;

    my ($invalid, $failedFilters) = (0, 0);

    my ($chr, @fields, %wantedData, $start, $end, $wantedStart, $wantedEnd);
    FH_LOOP: while( my $line = $fh->getline() ) {
      chomp $line;
      @fields = split('\t', $line);

      if(! $self->_validLine(\@fields, $., $reqIdxHref, $numColumns) ) {
        $invalid++;
        next FH_LOOP;
      }

      if(! $self->_passesFilter($fieldsToFilterOnIdx, \@fields, $.)) {
        $failedFilters++;
        next FH_LOOP;
      }

      $self->_transform($fieldsToTransformIdx, \@fields);

      # Transforms $chr if it's not prepended with a 'chr' or is 'chrMT' or 'MT'
      # and checks against our list of wanted chromosomes
      $chr = $self->normalizedWantedChr( $fields[ $reqIdxHref->{$self->chromField} ] );

      if(!$chr || $chr ne $wantedChr) {
        if($self->chrPerFile) {
          # Because this is not an unusual occurance; there is only 1 chr wanted
          # and the function is called once for each chromoosome, we use debug
          # to reduce log clutter
          $self->log('debug', $self->name . "join track: chrs in file $file not wanted . Skipping");

          last FH_LOOP;
        }

        next FH_LOOP;
      }

      ($start, $end) = $self->_getPositions(\@fields, $reqIdxHref);

      %wantedData = ();
      FNAMES_LOOP: for my $name (keys %$featureIdxHref) {
        my $value = $self->coerceFeatureType( $name, $fields[ $featureIdxHref->{$name} ] );

        $wantedData{$name} = $value;
      }

      for (my $i = 0; $i < @$wantedPositionsAref; $i++) {
        $wantedStart = $wantedPositionsAref->[$i][0];
        $wantedEnd = $wantedPositionsAref->[$i][1];

        # The join tracks accumulate a large amount of useless (for my current use case) information
        # namely, all of the single nucleotide variants that are already reported for a given position
        # The real use of the join track currently is to report all of the really large variants when they
        # overlap a gene, so let's do just that, by check against our maxVariantSize
        if( ( ($start >= $wantedStart && $start <= $wantedEnd) || ($end >= $wantedStart && $end <= $wantedEnd) ) &&
        $end + 1 - $start > $self->maxVariantSize) {
          &$callback(\%wantedData, $i);
          undef %wantedData;
        }
      }
    }

    if(!close($fh) && $? != 13) {
      $self->log('fatal', $self->name . " join track: failed to close $file with $! ($?)");
      die $self->name . " join track: failed to close $file with $!";
    } else {
      $self->log('info', $self->name . " join track: closed $file with $?");
      $self->log('info', $self->name . " join track: invalid lines found while joining on $file: $invalid");
      $self->log('info', $self->name . " join track: lines that didn't pass filters while joining on $file: $failedFilters");
    }
  }

  $self->log('info', $self->name . " join track: finished for $wantedChr");
}

sub _getHeaderFields {
  my ($self, $file, $firstLine, $wantedFeaturesAref) = @_;

  my @requiredFields = ($self->chromField, $self->chromStartField, $self->chromEndField);

  chomp $firstLine;

  # If the user wanted to transform the input field names, do, so source field names match
  # those expected by the track
  my @fields = map{ $self->fieldMap->{$_} || $_ } split('\t', $firstLine);

  my $numColumns = @fields;

  my %featureIdx;
  my %reqIdx;
  my %fieldsToTransformIdx;
  my %fieldsToFilterOnIdx;

  # Which fields are required (chrom, chromStart, chromEnd)
  REQ_LOOP: for my $field (@requiredFields) {
    my $idx = firstidx {$_ eq  $field} @fields; #returns -1 if not found

    if($idx >  -1) { #bitwise complement, makes -1 0
      $reqIdx{$field} = $idx;
      next REQ_LOOP; #label for clarity
    }

    $self->log('fatal', $self->name . ": required field $field missing in $file header");
    die $self->name . ": required field $field missing in $file header";
  }

  # Which fields the user specified under "features" key in config file
  FEATURE_LOOP: for my $fname (@$wantedFeaturesAref) {
    my $idx = firstidx {$_ eq $fname} @fields;

    if($idx > -1) { #only non-0 when non-negative, ~0 > 0
      $featureIdx{ $fname } = $idx;
      next FEATURE_LOOP;
    }

    $self->log('fatal', $self->name . ": feature $fname missing in $file header");
    die $self->name . ": feature $fname missing in $file header";
  }

  # Which fields user wants to filter the value of against some config-defined value
  FILTER_LOOP: for my $fname ($self->allFieldsToFilterOn) {
    my $idx = firstidx {$_ eq $fname} @fields;

    if($idx > -1) { #only non-0 when non-negative, ~0 > 0
      $fieldsToFilterOnIdx{ $fname } = $idx;
      next FILTER_LOOP;
    }

    $self->log('fatal', $self->name . ": feature $fname missing in $file header");
    die $self->name . ": feature $fname missing in $file header";
  }

  # Which fields user wants to modify the values of in a config-defined way
  TRANSFORM_LOOP: for my $fname ($self->allFieldsToTransform) {
    my $idx = firstidx {$_ eq $fname} @fields;

    if($idx > -1) { #only non-0 when non-negative, ~0 > 0
      $fieldsToTransformIdx{ $fname } = $idx;
      next TRANSFORM_LOOP;
    }

    $self->log('fatal', $self->name . ": feature $fname missing in $file header");
    die $self->name . ": feature $fname missing in $file header";
  }

  return (\%featureIdx, \%reqIdx, \%fieldsToTransformIdx, \%fieldsToFilterOnIdx, $numColumns);
}

# TODO: think about adding back dubg logging for _validLine, _passesFilter
sub _validLine {
  my ($self, $fieldAref, $lineNumber, $reqIdxHref, $numColumns) = @_;

  if(@$fieldAref != $numColumns) {
    # $self->log('debug', "Line $lineNumber has fewer columns than expected, skipping");
    return;
  }

  # Some files are misformatted, ex: clinvar's tab delimited
  if( !looks_like_number( $fieldAref->[ $reqIdxHref->{$self->chromStartField} ] )
  || !looks_like_number(  $fieldAref->[ $reqIdxHref->{$self->chromEndField} ] ) ) {
    # $self->log('debug', "Line $lineNumber Start or stop doesn't look like a number, skipping");
    return;
  }

  return 1;
}

sub _transform {
  my ($self, $fieldsToTransformIdx, $fieldsAref) = @_;
  #If the user wants to modify the values of any fields, do that first
  for my $fieldName ($self->allFieldsToTransform) {
    $fieldsAref->[ $fieldsToTransformIdx->{$fieldName} ] = 
      $self->transformField($fieldName, $fieldsAref->[ $fieldsToTransformIdx->{$fieldName} ] );
  }
}

sub _passesFilter {
  my ($self, $fieldsToFilterOnIdx, $fieldsAref, $lineNumber) = @_;
  # Then, if the user wants to exclude rows that don't pass some criteria
  # that they defined in the YAML file, allow that.
  for my $fieldName ($self->allFieldsToFilterOn) {
    if(!$self->passesFilter($fieldName, $fieldsAref->[ $fieldsToFilterOnIdx->{$fieldName} ] ) ) {
      # $self->log('debug', "Line $lineNumber $fieldName doesn't pass filter: $fieldsAref->[ $fieldsToFilterOnIdx->{$fieldName} ]");
      return;
    }
  }
  return 1;
}

sub _getPositions {
  my ($self, $fieldsAref, $reqIdxHref) = @_;

  my $start = $fieldsAref->[ $reqIdxHref->{$self->chromStartField} ];
  my $end = $fieldsAref->[ $reqIdxHref->{$self->chromEndField} ];

  #From UCSC clinvar to bed http://genecats.cse.ucsc.edu/git-reports-history/v311/review/user/max/full/src/hg/utils/otto/clinvar/clinVarToBed.32cc9617debc808eb02eeaba28a8be3b705cd0dc.html
  # https://ideone.com/IOYReQ
  if($start > $end) {
    my $warn = $self->name . ": $reqIdxHref->{$self->chromField} ] $start > $end. Flipping chromStart and chromEnd.";
    $self->log('warn', $warn);

    ($start, $end) = ($end, $start);
  }

  # This is an insertion; the only case when start should == stop (for 0-based coordinates)
  if($start == $end) {
    $start = $end = $start - $self->based;
  } else { 
    #it's a normal change, or a deletion
    #0-based files are expected to be half-closed format, so subtract 1 from end 
    $start = $start - $self->based;
    $end = $end - $self->based - $self->_halfClosedOffset;
  }

  return ($start, $end);
}
__PACKAGE__->meta->make_immutable;

1;
