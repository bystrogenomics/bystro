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

use Seq::DBManager;
use Path::Tiny;
use Scalar::Util qw/looks_like_number/;

extends 'Seq::Base';

# We  add a few of our own annotation attributes
# These will be re-used in the body of the annotation processor below
# Users may configure these
has input_file => (is => 'rw', isa => AbsFile, coerce => 1, required => 1,
  handles  => { inputFilePath => 'stringify' }, writer => 'setInputFile');

# Maximum (signed) size of del allele
has maxDel => (is => 'ro', isa => 'Int', default => -32, writer => 'setMaxDel');

# TODO: formalize: check that they have name and args properties
has fileProcessors => (is => 'ro', isa => 'HashRef', default => 'bystro-vcf');

# Defines most of the properties that can be configured at run time
# Needed because there are variations of Seq.pm, ilke SeqFromQuery.pm
# Requires logPath to be provided (currently found in Seq::Base)
with 'Seq::Definition', 'Seq::Role::Validator';

# To initialize Seq::Base with only getters
has '+readOnly' => (init_arg => undef, default => 1);

# TODO: further reduce complexity
sub BUILD {
  my $self = shift;

  if($self->maxDel > 0) {
    $self->setMaxDel(-$self->maxDel);
  }

  ################## Make the full output path ######################
  # The output path always respects the $self->output_file_base attribute path;
  $self->{_outPath} = $self->_workingDir->child($self->outputFilesInfo->{annotation});

  # Must come before statistics, which relies on a configured Seq::Tracks
  #Expects DBManager to have been given a database_dir
  $self->{_db} = Seq::DBManager->new();

  # Initializes the tracks
  # This ensures that the tracks are configured, and headers set
  $self->{_tracks} = $self->tracksObj;
}

sub annotate {
  my $self = shift;

  $self->log( 'info', 'Checking input file format' );


  # TODO: For now we only accept tab separated files
  # We could change this, although comma separation causes may cause with our fields
  # And is slower, since we cannot split on a constant
  # We would also need to take care with properly escaping intra-field commas
  # $err = $self->setDelimiter($firstLine);

  # if($err) {
  #   $self->_errorWithCleanup($err);
  #   return ($err, undef);
  # }

  # Calling in annotate allows us to error early
  my $err;
  ($err, $self->{_chunkSize}) = $self->getChunkSize($self->input_file, $self->maxThreads);

  if($err) {
    $self->_errorWithCleanup($err);
    return ($err, undef);
  }

  $self->log('debug', "chunk size is $self->{_chunkSize}");

  #################### Validate the input file ################################
  # Converts the input file if necessary
  ($err, my $fileType) = $self->validateInputFile($self->input_file);

  if($err) {
    $self->_errorWithCleanup($err);
    return ($err, undef);
  }

  # TODO: Handle Sig Int (Ctrl + C) to close db, clean up temp dir
  # local $SIG{INT} = sub {
  #   my $message = shift;
  # };

  if($fileType eq 'snp') {
    $self->log( 'info', 'Beginning annotation' );
    return $self->annotateFile('snp');
  }

  if($fileType eq 'vcf') {
    $self->log( 'info', 'Beginning annotation' );
    return $self->annotateFile('vcf');
  }

  # TODO: we don't really check for valid vcf, just assume it is
  # So this message is never reached
  $self->_errorWithCleanup("File type isn\'t vcf or snp. Please use one of these files");
  return ("File type isn\'t vcf or snp. Please use one of these files", undef);
}

sub annotateFile {
  #Inspired by T.S Wingo: https://github.com/wingolab-org/GenPro/blob/master/bin/vcfToSnp
  my $self = shift;
  my $type = shift;

  my ($err, $fh, $outFh, $statsFh) = $self->_getFileHandles($type);

  if($err) {
    $self->_errorWithCleanup($err);
    return ($err, undef);
  }

  ########################## Write the header ##################################
  my $header = <$fh>;
  $self->setLineEndings($header);

  my $finalHeader = $self->_getFinalHeader($header);
  my $outputHeader = $finalHeader->getString();

  say $outFh $outputHeader;

  if($statsFh) {
    say $statsFh $outputHeader;
  }

  # Now that header is prepared, make the outputter
  my $outputter = Seq::Output->new({header => $finalHeader});

  ######################## Build the fork pool #################################
  my $abortErr;

  # Report every 1e4 lines, to avoid thrashing receiver
  my $progressFunc = $self->makeLogProgressAndPrint(\$abortErr, $outFh, $statsFh, 2e4);
  MCE::Loop::init {
    max_workers => $self->maxThreads || 8, use_slurpio => 1,
    # bystro-vcf outputs a very small row; fully annotated through the alt column (-ref -discordant)
    # so accumulate less than we would if processing full .snp
    chunk_size => $self->{_chunkSize} > 8192 ? "8192K" : $self->{_chunkSize}. "K",
    gather => $progressFunc,
  };

  # We separate out the reference track getter so that we can check for discordant
  # bases, and pass the true reference base to other getters that may want it (like CADD)

  # To avoid the Moose/Mouse accessor penalty, store reference to underlying data
  my $db = $self->{_db};
  my $refTrackGetter = $self->tracksObj->getRefTrackGetter();
  my @trackGettersExceptReference = @{$self->tracksObj->getTrackGettersExceptReference()};

  my $trackIndices = $finalHeader->getParentIndices();
  my $refTrackIdx = $trackIndices->{$refTrackGetter->name};

  my %wantedChromosomes = %{ $refTrackGetter->chromosomes };
  my $maxDel = $self->maxDel;

  # This is going to be copied on write... avoid a bunch of function calls
  # Each thread will get its own %cursors object
  my %cursors = ();

  # TODO: don't annotate MT (GRCh37) if MT not explicitly specified
  # to avoid issues between GRCh37 and hg19 chrM vs MT
  # my %normalizedNames = %{$self->normalizedWantedChrs};

  mce_loop_f {
    #my ($mce, $slurp_ref, $chunk_id) = @_;
    #    $_[0], $_[1],     $_[2]
    #open my $MEM_FH, '<', $slurp_ref; binmode $MEM_FH, ':raw';
    open my $MEM_FH, '<', $_[1]; binmode $MEM_FH, ':raw';

    my $total = 0;

    my @indelDbData;
    my @indelRef;
    my @lines;
    my $dataFromDbAref;
    my $zeroPos;
    # Each line is expected to be
    # chrom \t pos \t type \t inputRef \t alt \t hets \t homozygotes \n
    # the chrom is always in ucsc form, chr (the golang program guarantees it)

    while (my $line = $MEM_FH->getline()) {
      chomp $line;

      my @fields = split '\t', $line;
      $total++;

      if(!$wantedChromosomes{$fields[0]}) {
        next;
      }

      $zeroPos = $fields[1] - 1;

      # Caveat: It seems that, per database ($chr), we can have only one
      # read-only transaction; so ... yeah can't combine with dbRead, dbReadOne
      if(!$cursors{$fields[0]}) {
        $cursors{$fields[0]} = $db->dbStartCursorTxn($fields[0]);
      }

      $dataFromDbAref = $db->dbReadOneCursorUnsafe($cursors{$fields[0]}, $zeroPos);

      if(!defined $dataFromDbAref) {
        $self->_errorWithCleanup("Wrong assembly? $fields[0]\: $fields[1] not found.");
        # Store a reference to the error, allowing us to exit with a useful fail message
        MCE->gather(0, 0, "Wrong assembly? $fields[0]\: $fields[1] not found.");
        $_[0]->abort();
        return;
      }

      if(length($fields[4]) > 1) {
        # INS or DEL
        if(looks_like_number($fields[4])) {
          # We ignore -1 alleles, treat them just like SNPs
          if($fields[4] < -1)  {
            # Grab everything from + 1 the already fetched position to the $pos + number of deleted bases - 1
            # Note that position_1_based - (negativeDelLength + 2) == position_0_based + (delLength - 1)
            if($fields[4] < $maxDel) {
              @indelDbData = ($fields[1] .. $fields[1] - ($maxDel + 2));
              # $self->log('info', "$fields[0]:$fields[1]: long deletion. Annotating up to $maxDel");
            } else {
              @indelDbData = ($fields[1] .. $fields[1] - ($fields[4] + 2));
            }

            #last argument: skip commit
            $db->dbReadCursorUnsafe($cursors{$fields[0]},  \@indelDbData);

            #Note that the first position keeps the same $inputRef
            #This means in the (rare) discordant multiallelic situation, the reference
            #Will be identical between the SNP and DEL alleles
            #faster than perl-style loop (much faster than c-style)
            @indelRef = ($fields[3], map { $refTrackGetter->get($_) } @indelDbData);

            #Add the db data that we already have for this position
            unshift @indelDbData, $dataFromDbAref;
          }
        } else {
          #It's an insertion, we always read + 1 to the position being annotated
          # which itself is + 1 from the db position, so we read  $out[1][0][0] to get the + 1 base
          # Read without committing by using 1 as last argument
          @indelDbData = ($dataFromDbAref, $db->dbReadOneCursorUnsafe($cursors{$fields[0]}, $fields[1]));

          #Note that the first position keeps the same $inputRef
          #This means in the (rare) discordant multiallelic situation, the reference
          #Will be identical between the SNP and DEL alleles
          @indelRef = ($fields[3], $refTrackGetter->get($indelDbData[1]));
        }
      }

      if(@indelDbData) {
        ############### Gather all track data (besides reference) #################
        for my $posIdx (0 .. $#indelDbData) {
          for my $track (@trackGettersExceptReference) {
            $fields[$trackIndices->{$track->name}] //= [];

            $track->get($indelDbData[$posIdx], $fields[0], $indelRef[$posIdx], $fields[4], $posIdx,
              $fields[$trackIndices->{$track->name}], $zeroPos + $posIdx);
          }

          $fields[$refTrackIdx][$posIdx] = $indelRef[$posIdx];
        }

        # If we have multiple indel alleles at one position, need to clear stored values
        @indelDbData = ();
        @indelRef = ();
      } else {
        for my $track (@trackGettersExceptReference) {
          $fields[$trackIndices->{$track->name}] //= [];
          $track->get($dataFromDbAref, $fields[0], $fields[3], $fields[4], 0,
            $fields[$trackIndices->{$track->name}], $zeroPos);
        }

        $fields[$refTrackIdx][0] = $refTrackGetter->get($dataFromDbAref);
      }

       # 3 holds the input reference, we'll replace this with the discordant status
      $fields[3] = $refTrackGetter->get($dataFromDbAref) ne $fields[3] ? 1 : 0;

      push @lines, \@fields;
    }

    close $MEM_FH;

    if(@lines) {
      MCE->gather(scalar @lines, $total - @lines, undef, $outputter->makeOutputString(\@lines));
    } else {
      MCE->gather(0, $total);
    }

  } $fh;

  # Force flush
  $progressFunc->(0, 0, undef, undef, 1);

  MCE::Loop::finish();

  # Unfortunately, MCE::Shared::stop() removes the value of $abortErr
  # according to documentation, and I did not see mention of a way
  # to copy the data from a scalar, and don't want to use a hash for this alone
  # So, using a scalar ref to abortErr in the gather function.
  if($abortErr) {
    say "Aborted job due to $abortErr";

    # Database & tx need to be closed
    $db->cleanUp();

    return ('Job aborted due to error', undef);
  }

  ################ Finished writing file. If statistics, print those ##########
  # Sync to ensure all files written
  # This simply tries each close/sync/move operation in order
  # And returns an error early, or proceeds to next operation
  $err = $self->safeClose($outFh)
          ||
          ($statsFh && $self->safeClose($statsFh))
          ||
          $self->safeSystem('sync')
          ||
          $self->_moveFilesToOutputDir();

  if($err) {
    $self->_errorWithCleanup($err);
    return ($err, undef);
  }

  $db->cleanUp();

  return ($err, $self->outputFilesInfo);
}

sub makeLogProgressAndPrint {
  my ($self, $abortErrRef, $outFh, $statsFh, $throttleThreshold) = @_;

  my $totalAnnotated = 0;
  my $totalSkipped = 0;

  my $publish = $self->hasPublisher;

  my $thresholdAnn = 0;
  my $thresholdSkipped = 0;

  if(!$throttleThreshold) {
    $throttleThreshold = 1e4;
  }
  return sub {
    #<Int>$annotatedCount, <Int>$skipCount, <Str>$err, <Str>$outputLines, <Bool> $forcePublish = @_;
    ##    $_[0],          $_[1]           , $_[2],     $_[3].           , $_[4]
    if($_[2]) {
      $$abortErrRef = $_[2];
      return;
    }

    if($publish) {
      $thresholdAnn += $_[0];
      $thresholdSkipped += $_[1];

      if($_[4] || $thresholdAnn + $thresholdSkipped >= $throttleThreshold) {
        $totalAnnotated += $thresholdAnn;
        $totalSkipped += $thresholdSkipped;

        $self->publishProgress($totalAnnotated, $totalSkipped);

        $thresholdAnn = 0;
        $thresholdSkipped = 0;
      }
    }

    if($_[3]) {
      if($statsFh) {
        print $statsFh $_[3];
      }

      print $outFh $_[3];
    }
  }
}

sub _getFileHandles {
  my ($self, $type) = @_;

  my ($outFh, $statsFh, $inFh);
  my $err;

  ($err, $inFh) = $self->_openAnnotationPipe($type);

  if($err) {
    return ($err,  undef, undef, undef);
  }

  if($self->run_statistics) {
    ########################## Tell stats program about our annotation ##############
    # TODO: error handling if fh fails to open
    my $statArgs = $self->_statisticsRunner->getStatsArguments();

    $err = $self->safeOpen($statsFh, "|-", $statArgs);

    if($err) {
      return ($err,  undef, undef, undef);
    }
  }

  # $fhs{stats} = $$statsFh;
  ($err, $outFh) = $self->getWriteFh($self->{_outPath});

  if($err) {
    return ($err, undef, undef, undef);
  }

  return (undef, $inFh, $outFh, $statsFh);
}

sub _openAnnotationPipe {
  my ($self, $type) = @_;

  my $errPath = $self->_workingDir->child($self->input_file->basename . '.file-log.log');

  my $inPath = $self->inputFilePath;
  my $echoProg = $self->isCompressedSingle($inPath) ? $self->gzip . ' ' . $self->decompressArgs : 'cat';

  if(!$self->fileProcessors->{$type}) {
    $self->_errorWithCleanup("No fileProcessors defined for $type file type");
  }

  my $fp = $self->fileProcessors->{$type};
  my $args = $fp->{program} . " " . $fp->{args};

  my $fh;

  for my $type (keys %{$self->outputFilesInfo}) {
    if(index($args, "\%$type") > -1) {
      substr($args, index($args, "\%$type"), length("\%$type"))
        = $self->_workingDir->child($self->outputFilesInfo->{$type});
    }
  }

  # TODO:  add support for GQ filtering in vcf
  my $err = $self->safeOpen($fh, '-|', "$echoProg $inPath | $args 2> $errPath");

  return ($err, $fh);
}

sub _getFinalHeader {
  my ($self, $header) = @_;
  ######### Build the header, and write it as the first line #############
  my $finalHeader = Seq::Headers->new();

  chomp $header;

  my @headerFields = split '\t', $header;

  # Our header class checks the name of each feature
  # It may be, more than likely, that the pre-processor names the 4th column 'ref'
  # We replace this column with trTv
  # This not only now reflects its actual function
  # but prevents name collision issues resulting in the wrong header idx
  # being generated for the ref track
  $headerFields[3] = $self->discordantField;

  # Bystro takes data from a file pre-processor, which spits out a common
  # intermediate format
  # This format is very flexible, in fact Bystro doesn't care about the output
  # of the pre-processor, provided that the following is found in the corresponding
  # indices:
  # idx 0: chromosome,
  # idx 1: position
  # idx 3: the reference (we rename this to discordant)
  # idx 4: the alternate allele

  # Prepend all of the headers created by the pre-processor
  $finalHeader->addFeaturesToHeader(\@headerFields, undef, 1);

  return $finalHeader;
}

sub _errorWithCleanup {
  my ($self, $msg) = @_;

  $self->log('error', $msg);

  $self->{_db}->cleanUp();

  return $msg;
}

__PACKAGE__->meta->make_immutable;

1;