use 5.10.0;
use strict;
use warnings;

package Seq;

our $VERSION = '0.001';
# ABSTRACT: Annotate a snp file

# TODO: make temp_dir handling more transparent
use Mouse 2;
use Types::Path::Tiny qw/AbsFile/;

use namespace::autoclean;

use MCE::Loop;

use Seq::InputFile;
use Seq::Output;
use Seq::Output::Delimiters;
use Seq::Headers;
use JSON::XS;
use Seq::DBManager;
use Path::Tiny;
use Scalar::Util qw/looks_like_number/;

extends 'Seq::Base';

# We  add a few of our own annotation attributes
# These will be re-used in the body of the annotation processor below
# Users may configure these
has input_file => ( is => 'rw', isa => 'Str', required => 1 );

# Maximum (signed) size of del allele
has maxDel => ( is => 'ro', isa => 'Int', default => -32, writer => 'setMaxDel' );

# TODO: formalize: check that they have name and args properties
has fileProcessors => ( is => 'ro', isa => 'HashRef', required => 1 );

# Defines most of the properties that can be configured at run time
# Requires logPath to be provided (currently found in Seq::Base)
with 'Seq::Definition', 'Seq::Role::Validator';

# To initialize Seq::Base with only getters
has '+readOnly' => ( init_arg => undef, default => 1 );

# TODO: further reduce complexity
sub BUILD {
  my $self = shift;

  if ( $self->maxDel > 0 ) {
    $self->setMaxDel( -$self->maxDel );
  }

  ################## Make the full output path ######################
  # The output path always respects the $self->output_file_base attribute path;
  $self->{_outPath} =
    $self->_workingDir->child( $self->outputFilesInfo->{annotation} );

  $self->{_headerPath} = $self->_workingDir->child( $self->outputFilesInfo->{header} );

  # Must come before statistics, which relies on a configured Seq::Tracks
  #Expects DBManager to have been given a database_dir
  $self->{_db} = Seq::DBManager->new();

  # When tracksObj accessor is called, a function is executed that results in the database being set to read only
  # when read only mode is called for (Seq::Base instantiated with readOnly => 1)
  # Therefore it is important to call tracksObj before any other database calls
  # to make sure that read-only semantics are enforced throughout the annotation process
  # To ensure that we don't accidentally defer setting database mode until child proceses begin to annotate (in which case they will race to set the database mode)
  # from hereon we will only use $self->{_tracks} to refer to tracks objects.
  $self->{_tracks} = $self->tracksObj;
}

sub annotate {
  my $self = shift;

  $self->log( 'info', 'Checking input file format' );

  my ( $err, $fileType ) = $self->validateInputFile( $self->input_file );

  if ($err) {
    $self->_errorWithCleanup($err);
    return ( $err, undef );
  }

  $self->log( 'info', 'Beginning annotation' );
  return $self->annotateFile($fileType);
}

sub annotateFile {
  #Inspired by T.S Wingo: https://github.com/wingolab-org/GenPro/blob/master/bin/vcfToSnp
  my $self = shift;
  my $type = shift;

  my ( $err, $fh, $outFh, $statsFh, $headerFh ) = $self->_getFileHandles($type);

  if ($err) {
    $self->_errorWithCleanup($err);
    return ( $err, undef );
  }

  ########################## Write the header ##################################
  my $header = <$fh>;
  $self->setLineEndings($header);

  my ( $finalHeader, $numberSplitFields ) = $self->_getFinalHeader($header);

  ## A programmatically useful representation of the header
  say $headerFh encode_json( $finalHeader->getOrderedHeader() );
  my $outputHeader = $finalHeader->getString();

  if ( !$self->outputJson ) {
    say $outFh $outputHeader;
  }

  if ($statsFh) {
    say $statsFh $outputHeader;
  }

  ######################## Build the fork pool #################################
  my $abortErr;

  my $messageFreq = ( 2e4 / 4 ) * $self->maxThreads;

  # Report every 1e4 lines, to avoid thrashing receiver
  my $progressFunc =
    $self->makeLogProgressAndPrint( \$abortErr, $outFh, $statsFh, $messageFreq );
  MCE::Loop::init {
    max_workers => $self->maxThreads || 8,
    use_slurpio => 1,
    chunk_size  => 'auto',
    gather      => $progressFunc,
  };

  # We separate out the reference track getter so that we can check for discordant
  # bases, and pass the true reference base to other getters that may want it (like CADD)

  # To avoid the Moose/Mouse accessor penalty, store reference to underlying data
  my $db             = $self->{_db};
  my $refTrackGetter = $self->{_tracks}->getRefTrackGetter();
  my @trackGettersExceptReference =
    @{ $self->{_tracks}->getTrackGettersExceptReference() };
  my @trackIndicesExceptReference = 0 .. $#trackGettersExceptReference;

  my $outIndicesMap = $finalHeader->getParentIndices();

  my @outIndicesExceptReference =
    map { $outIndicesMap->{ $_->name } } @trackGettersExceptReference;

  ######### Set Outputter #########
  my @allOutIndices =
    map { $outIndicesMap->{ $_->name } } @{ $self->{_tracks}->trackGetters };

  # Now that header is prepared, make the outputter
  # Note, that the only features that we need to iterate over
  # Are the features that come from our database
  # Meaning, we can skip anything forwarded from the pre-processor

  my $outputter = Seq::Output->new(
    {
      header          => $finalHeader,
      trackOutIndices => \@allOutIndices,
      refTrackName    => $refTrackGetter->name
    }
  );

  ###### Processes pre-processor output passed from file reader/producer #######
  my $discordantIdx  = $outIndicesMap->{ $self->discordantField };
  my $refTrackOutIdx = $outIndicesMap->{ $refTrackGetter->name };

  #Accessors are amazingly slow; it takes as long to call ->name as track->get
  #after accounting for the nubmer of calls to ->name
  my %wantedChromosomes = %{ $refTrackGetter->chromosomes };
  my $maxDel            = $self->maxDel;

  my $outJson = $self->outputJson;

  mce_loop_f {
    #my ($mce, $slurp_ref, $chunk_id) = @_;
    #    $_[0], $_[1],     $_[2]
    open my $MEM_FH, '<', $_[1];
    binmode $MEM_FH, ':raw';

    my $total = 0;

    my @indelDbData;
    my @indelRef;
    my @lines;
    my $dataFromDbAref;
    my $zeroPos;

    # This is going to be copied on write... avoid a bunch of function calls
    # Each thread will get its own %cursors object
    # But start in child because relying on COW seems like it could lead to
    # future bugs (in, say Rust if sharing between user threads)
    my %cursors = ();

    # Each line is expected to be
    # chrom \t pos \t type \t inputRef \t alt \t hets \t homozygotes \n
    # the chrom is always in ucsc form, chr (the golang program guarantees it)
    my $outputJson = $self->outputJson;
    while ( my $line = $MEM_FH->getline() ) {
      chomp $line;

      my @fields = split( '\t', $line, $numberSplitFields );

      $total++;

      if ( !$wantedChromosomes{ $fields[0] } ) {
        next;
      }

      $zeroPos = $fields[1] - 1;

      # Caveat: It seems that, per database ($chr), we can have only one
      # read-only transaction; so ... yeah can't combine with dbRead, dbReadOne
      if ( !$cursors{ $fields[0] } ) {
        $cursors{ $fields[0] } = $db->dbStartCursorTxn( $fields[0] );
      }

      $dataFromDbAref = $db->dbReadOneCursorUnsafe( $cursors{ $fields[0] }, $zeroPos );

      if ( !defined $dataFromDbAref ) {
        $self->_errorWithCleanup("Wrong assembly? $fields[0]\: $fields[1] not found.");
        # Store a reference to the error, allowing us to exit with a useful fail message
        MCE->gather( 0, 0, "Wrong assembly? $fields[0]\: $fields[1] not found." );
        $_[0]->abort();
        return;
      }

      if ( length( $fields[4] ) > 1 ) {
        # INS or DEL
        if ( looks_like_number( $fields[4] ) ) {
          # We ignore -1 alleles, treat them just like SNPs
          if ( $fields[4] < -1 ) {
            # Grab everything from + 1 the already fetched position to the $pos + number of deleted bases - 1
            # Note that position_1_based - (negativeDelLength + 2) == position_0_based + (delLength - 1)
            if ( $fields[4] < $maxDel ) {
              @indelDbData = ( $fields[1] .. $fields[1] - ( $maxDel + 2 ) );
            }
            else {
              @indelDbData = ( $fields[1] .. $fields[1] - ( $fields[4] + 2 ) );
            }

            #last argument: skip commit
            $db->dbReadCursorUnsafe( $cursors{ $fields[0] }, \@indelDbData );

            #Note that the first position keeps the same $inputRef
            #This means in the (rare) discordant multiallelic situation, the reference
            #Will be identical between the SNP and DEL alleles
            #faster than perl-style loop (much faster than c-style)
            @indelRef = ( $fields[3], map { $refTrackGetter->get($_) } @indelDbData );

            #Add the db data that we already have for this position
            unshift @indelDbData, $dataFromDbAref;
          }
        }
        else {
          #It's an insertion, we always read + 1 to the position being annotated
          # which itself is + 1 from the db position, so we read  $out[1][0][0] to get the + 1 base
          # Read without committing by using 1 as last argument
          @indelDbData = (
            $dataFromDbAref, $db->dbReadOneCursorUnsafe( $cursors{ $fields[0] }, $fields[1] )
          );

          #Note that the first position keeps the same $inputRef
          #This means in the (rare) discordant multiallelic situation, the reference
          #Will be identical between the SNP and DEL alleles
          @indelRef = ( $fields[3], $refTrackGetter->get( $indelDbData[1] ) );
        }
      }

      if (@indelDbData) {
        ############### Gather all track data (besides reference) #################
        for my $posIdx ( 0 .. $#indelDbData ) {
          for my $trackIndex (@trackIndicesExceptReference) {
            $fields[ $outIndicesExceptReference[$trackIndex] ] //= [];

            $trackGettersExceptReference[$trackIndex]->get(
              $indelDbData[$posIdx], $fields[0], $indelRef[$posIdx], $fields[4], $posIdx,
              $fields[ $outIndicesExceptReference[$trackIndex] ],
              $zeroPos + $posIdx
            );
          }

          $fields[$refTrackOutIdx][$posIdx] = $indelRef[$posIdx];
        }

        # If we have multiple indel alleles at one position, need to clear stored values
        @indelDbData = ();
        @indelRef    = ();
      }
      else {
        for my $trackIndex (@trackIndicesExceptReference) {
          $fields[ $outIndicesExceptReference[$trackIndex] ] //= [];

          $trackGettersExceptReference[$trackIndex]
            ->get( $dataFromDbAref, $fields[0], $fields[3], $fields[4], 0,
            $fields[ $outIndicesExceptReference[$trackIndex] ], $zeroPos );
        }

        $fields[$refTrackOutIdx][0] = $refTrackGetter->get($dataFromDbAref);
      }

      # 3 holds the input reference, we'll replace this with the discordant status
      $fields[$discordantIdx] =
        $refTrackGetter->get($dataFromDbAref) ne $fields[3] ? 1 : 0;

      push @lines, \@fields;
    }

    close $MEM_FH;

    if (@lines) {
      if ($outJson) {
        MCE->gather( scalar @lines, $total - @lines, undef, encode_json( \@lines ) );
      }
      else {
        MCE->gather(
          scalar @lines,
          $total - @lines,
          undef, $outputter->makeOutputString( \@lines )
        );
      }
    }
    else {
      MCE->gather( 0, $total );
    }

  }
  $fh;

  # Force flush
  $progressFunc->( 0, 0, undef, undef, 1 );

  MCE::Loop::finish();

  # Unfortunately, MCE::Shared::stop() removes the value of $abortErr
  # according to documentation, and I did not see mention of a way
  # to copy the data from a scalar, and don't want to use a hash for this alone
  # So, using a scalar ref to abortErr in the gather function.
  if ($abortErr) {
    say "Aborted job due to $abortErr";

    # Database & tx need to be closed
    $db->cleanUp();

    return ( 'Job aborted due to error', undef );
  }

  ################ Finished writing file. If statistics, print those ##########
  # Sync to ensure all files written
  # This simply tries each close/sync/move operation in order
  # And returns an error early, or proceeds to next operation
  my $configOutPath = $self->_workingDir->child( $self->outputFilesInfo->{config} );

  $err =
       $self->safeClose($outFh)
    || ( $statsFh && $self->safeClose($statsFh) )
    || $self->safeSystem( "cp " . $self->config . " $configOutPath" )
    || $self->safeSystem('sync')
    || $self->_moveFilesToOutputDir();

  if ($err) {
    $self->_errorWithCleanup($err);
    return ( $err, undef );
  }

  $db->cleanUp();

  return ( $err, $self->outputFilesInfo );
}

sub makeLogProgressAndPrint {
  my ( $self, $abortErrRef, $outFh, $statsFh, $throttleThreshold ) = @_;

  my $totalAnnotated = 0;
  my $totalSkipped   = 0;

  my $publish = $self->hasPublisher;

  my $thresholdAnn     = 0;
  my $thresholdSkipped = 0;

  if ( !$throttleThreshold ) {
    $throttleThreshold = 1e4;
  }
  return sub {
    #<Int>$annotatedCount, <Int>$skipCount, <Str>$err, <Str>$outputLines, <Bool> $forcePublish = @_;
    ##    $_[0],          $_[1]           , $_[2],     $_[3].           , $_[4]
    if ( $_[2] ) {
      $$abortErrRef = $_[2];
      return;
    }

    if ($publish) {
      $thresholdAnn     += $_[0];
      $thresholdSkipped += $_[1];

      if ( $_[4] || $thresholdAnn + $thresholdSkipped >= $throttleThreshold ) {
        $totalAnnotated += $thresholdAnn;
        $totalSkipped   += $thresholdSkipped;

        $self->publishProgress( $totalAnnotated, $totalSkipped );

        $thresholdAnn     = 0;
        $thresholdSkipped = 0;
      }
    }

    if ( $_[3] ) {
      if ($statsFh) {
        print $statsFh $_[3];
      }

      print $outFh $_[3];
    }
  }
}

sub _getFileHandles {
  my ( $self, $type ) = @_;

  my ( $outFh, $statsFh, $inFh, $headerFh );
  my $err;

  ( $err, $inFh ) = $self->_openAnnotationPipe($type);

  if ($err) {
    return ( $err, undef, undef, undef );
  }

  if ( $self->run_statistics ) {
    ########################## Tell stats program about our annotation ##############
    # TODO: error handling if fh fails to open
    my $statArgs = $self->_statisticsRunner->getStatsArguments();

    $err = $self->safeOpen( $statsFh, "|-", $statArgs );

    if ($err) {
      return ( $err, undef, undef, undef );
    }
  }

  # $fhs{stats} = $$statsFh;
  ( $err, $outFh ) = $self->getWriteFh( $self->{_outPath} );

  if ($err) {
    return ( $err, undef, undef, undef );
  }

  ( $err, $headerFh ) = $self->getWriteFh( $self->{_headerPath} );

  if ($err) {
    return ( $err, undef, undef, undef );
  }

  return ( undef, $inFh, $outFh, $statsFh, $headerFh );
}

sub _preparePreprocessorProgram {
  my ( $self, $type ) = @_;

  if ( !$self->fileProcessors->{$type} ) {
    $self->_errorWithCleanup("No fileProcessors defined for $type file type");
  }

  my $inPath   = $self->input_file;
  my $basename = path($inPath)->basename;

  my $errPath = $self->_workingDir->child( $basename . '.file-log.log' );

  #cat is wasteful, but we expect no one reads large uncompressed files
  my $echoProg = $self->getReadArgs($inPath) || "cat $inPath";

  my $fp = $self->fileProcessors->{$type};

  my $finalProgram;
  if ( $fp->{no_stdin} ) {
    $finalProgram = $fp->{program} . " " . $inPath;
  }
  else {
    $finalProgram = $echoProg . " | " . $fp->{program};
  }

  if ( $fp->{args} ) {
    my $args = $fp->{args};

    for my $type ( keys %{ $self->outputFilesInfo } ) {
      if ( index( $args, "\%$type\%" ) > -1 ) {
        substr( $args, index( $args, "\%$type\%" ), length("\%$type\%") ) =
          $self->_workingDir->child( $self->outputFilesInfo->{$type} );
      }
    }

    $finalProgram .= " $args";
  }

  return ( $finalProgram, $errPath );
}

sub _openAnnotationPipe {
  my ( $self, $type ) = @_;

  my ( $finalProgram, $errPath ) = $self->_preparePreprocessorProgram($type);

  my $fh;
  my $err = $self->safeOpen( $fh, '-|', "$finalProgram 2> $errPath" );

  return ( $err, $fh );
}

sub _getFinalHeader {
  my ( $self, $header ) = @_;
  chomp $header;

  ######### Build the header, and write it as the first line #############
  my $finalHeader = Seq::Headers->new();

  # Bystro takes data from a file pre-processor, which spits out a common
  # intermediate format
  # This format is very flexible, in fact Bystro doesn't care about the output
  # of the pre-processor, provided that the following is found in the corresponding
  # indices:
  # idx 0: chromosome,
  # idx 1: position
  # idx 3: the reference (we rename this to discordant)
  # idx 4: the alternate allele
  # idx 5 on: variable: anything the preprocessor provides
  my $numberSplitFields;
  my @headerFields;
  if ( $self->outputJson ) {
    @headerFields      = split( '\t', $header );
    $numberSplitFields = @headerFields;
  }
  else {
    # Avoid unnecessary work splitting parts of the file we will not be extracting individual fields from
    $numberSplitFields = 5 + 1;
    @headerFields      = split( '\t', $header, $numberSplitFields );
  }

  # We need to ensure that the ref field of the pre-processor is renamed
  # so to not conflict with the ref field of the reference track
  # because we store field names in a hash
  $headerFields[3] = $self->inputRefField;

  # Our header class checks the name of each feature
  # It may be, more than likely, that the pre-processor names the 4th column 'ref'
  # We replace this column with trTv
  # This not only now reflects its actual function
  # but prevents name collision issues resulting in the wrong header idx
  # being generated for the ref track
  push @headerFields, $self->discordantField;

  # Prepend all of the headers created by the pre-processor
  $finalHeader->addFeaturesToHeader( \@headerFields, undef, 1 );

  return ( $finalHeader, $numberSplitFields );
}

sub _errorWithCleanup {
  my ( $self, $msg ) = @_;

  $self->log( 'error', $msg );

  $self->{_db}->cleanUp();

  return $msg;
}

__PACKAGE__->meta->make_immutable;

1;
