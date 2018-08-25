use 5.10.0;
use strict;
use warnings;

package Seq;

our $VERSION = '0.001';

# ABSTRACT: Annotate a snp file

# TODO: make temp_dir handling more transparent
use Mouse 2;
use Types::Path::Tiny qw/AbsPath AbsFile AbsDir/;

use namespace::autoclean;

use DDP;

use MCE::Loop;

use Seq::InputFile;
use Seq::Output;
use Seq::Headers;

use Seq::DBManager;
use Path::Tiny;
use File::Which qw/which/;
use Carp qw/croak/;
use Scalar::Util qw/looks_like_number/;
use List::MoreUtils qw/first_index/;

use Cpanel::JSON::XS;

extends 'Seq::Base';

# We  add a few of our own annotation attributes
# These will be re-used in the body of the annotation processor below
# Users may configure these
has input_file => (is => 'rw', isa => AbsFile, coerce => 1, required => 1,
  handles  => { inputFilePath => 'stringify' }, writer => 'setInputFile');

# Maximum (signed) size of del allele
has maxDel => (is => 'ro', isa => 'Int', default => -32, writer => 'setMaxDel');

has minGq => (is => 'ro', isa => 'Num', default => '.95');

has snpProcessor => (is => 'ro', isa => 'Str', default => 'bystro-snp');
has vcfProcessor => (is => 'ro', isa => 'Str', default => 'bystro-vcf');

# Defines most of the properties that can be configured at run time
# Needed because there are variations of Seq.pm, ilke SeqFromQuery.pm
# Requires logPath to be provided (currently found in Seq::Base)
with 'Seq::Definition', 'Seq::Role::Validator';

# To initialize Seq::Base with only getters
has '+gettersOnly' => (init_arg => undef, default => 1);

# TODO: further reduce complexity
sub BUILD {
  my $self = shift;

  if($self->maxDel > 0) {
    $self->setMaxDel(-$self->maxDel);
  }

  ########### Create DBManager instance, and instantiate track singletons #########
  # Must come before statistics, which relies on a configured Seq::Tracks
  #Expects DBManager to have been given a database_dir
  $self->{_db} = Seq::DBManager->new();

  # Set the lmdb database to read only, remove locking
  # We MUST make sure everything is written to the database by this point
  # Disable this if need to rebuild one of the meta tracks, for one run
  $self->{_db}->setReadOnly(1);

  # We separate out the reference track getter so that we can check for discordant
  # bases, and pass the true reference base to other getters that may want it (like CADD)
  # Store these references as hashes, to avoid accessor penalty
  $self->{_refTrackGetter} = $self->tracksObj->getRefTrackGetter();
  $self->{_trackGettersExceptReference} = $self->tracksObj->getTrackGettersExceptReference();

  ######### Build the header, and write it as the first line #############
  my $headers = Seq::Headers->new();

  # Bystro has a single pseudo track, that is always present, regardless of whether
  # any tracks exist
  # Note: Field order is required to stay in the follwoing order, because
  # current API allows use of constant as array index:
  # these to be configured:
  # idx 0:  $self->chromField,
  # idx 1: $self->posField,
  # idx 2: $self->typeField,
  # idx 3: $self->discordantField,
  # index 4: $self->altField,
  # index 5: $self->heterozygotesField,
  # index 6: $self->homozygotesField
  $headers->addFeaturesToHeader([
    #index 0
    $self->chromField,
    #index 1
    $self->posField,
    #index 2
    $self->typeField,
    #index 3
    $self->discordantField,
    #index 4
    $self->altField,
    #index 5
    $self->trTvField,
    #index 6
    $self->heterozygotesField,
    #index 7
    $self->heterozygosityField,
    #index 8
    $self->homozygotesField,
    #index 9
    $self->homozygosityField,
    #index 10
    $self->missingField,
    #index 11
    $self->missingnessField,
    #index 12
    $self->sampleMafField,
  ], undef, 1);

  $self->{_lastHeaderIdx} = $#{$headers->get()};

  $self->{_trackIdx} = $headers->getParentFeaturesMap();

  ################### Creates the output file handler #################
  # Used in makeAnnotationString
  $self->{_outputter} = Seq::Output->new();

  ################## Make the full output path ######################
  # The output path always respects the $self->output_file_base attribute path;
  $self->{_outPath} = $self->_workingDir->child($self->outputFilesInfo->{annotation});
}

sub annotate {
  my $self = shift;

  $self->log( 'info', 'Checking input file format' );

  my $err;
  my $fh = $self->get_read_fh($self->input_file);
  my $firstLine = <$fh>;

  $err = $self->setLineEndings($firstLine);

  if($err) {
    $self->_errorWithCleanup($err);
    return ($err, undef);
  }

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
  ($err, $self->{_chunkSize}) = $self->getChunkSize($self->input_file, $self->max_threads);

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

  my $errPath = $self->_workingDir->child($self->input_file->basename . '.vcf-log.log');
  my $inPath = $self->inputFilePath;
  my $echoProg = $self->isCompressedSingle($inPath) ? $self->gzip . ' ' . $self->decompressArgs : 'cat';
  my $delim = $self->{_outputter}->delimiters->emptyFieldChar;
  my $minGq = $self->minGq;

  my $fh;

  # TODO: add support for GQ filtering in vcf
  if ($type eq 'snp') {
    open($fh, '-|', "$echoProg $inPath | " . $self->snpProcessor . " --emptyField $delim --minGq $minGq 2> $errPath");
  } elsif($type eq 'vcf') {
    # Retruns chr, pos, homozygotes, heterozygotes, alt, ref in that order, tab delim
    open($fh, '-|', "$echoProg $inPath | " . $self->vcfProcessor . " --emptyField $delim 2> $errPath");
  } else {
    $self->_errorWithCleanup("annotateFiles only accepts snp and vcf types");
    return ("annotateFiles only accepts snp and vcf types", undef);
  }

  # If user specified a temp output path, use that
  my $outFh = $self->get_write_fh( $self->{_outPath} );

  ########################## Write the header ##################################
  my $headers = Seq::Headers->new();
  my $outputHeader = $headers->getString();

  say $outFh $outputHeader;

  ########################## Tell stats program about our annotation ##############
  # TODO: error handling if fh fails to open
  my $statsFh;
  if($self->run_statistics) {
    my $args = $self->_statisticsRunner->getStatsArguments();
    open($statsFh, "|-", $args);

    say $statsFh $outputHeader;
  }

  my $abortErr;

  # Report every 1e4 lines, to avoid thrashing receiver
  my $progressFunc = $self->makeLogProgressAndPrint(\$abortErr, $outFh, $statsFh, 2e4);
  MCE::Loop::init {
    max_workers => $self->max_threads || 8, use_slurpio => 1,
    # bystro-vcf outputs a very small row; fully annotated through the alt column (-ref -discordant)
    # so accumulate less than we would if processing full .snp
    chunk_size => $self->{_chunkSize} > 4192 ? "4192K" : $self->{_chunkSize}. "K",
    gather => $progressFunc,
  };

  my $trackIndices = $self->{_trackIdx};
  my $refTrackIdx = $self->{_trackIdx}{$self->{_refTrackGetter}->name};
  my @trackGettersExceptReference = @{$self->{_trackGettersExceptReference}};
  my %wantedChromosomes = %{ $self->{_refTrackGetter}->chromosomes };
  my $maxDel = $self->maxDel;

  my $err = $self->setLineEndings("\n");

  if($err) {
    $self->_errorWithCleanup($err);
    return ($err, undef);
  }

  my $header = <$fh>;

  mce_loop_f {
    #my ($mce, $slurp_ref, $chunk_id) = @_;
    #    $_[0], $_[1],     $_[2]
    #open my $MEM_FH, '<', $slurp_ref; binmode $MEM_FH, ':raw';
    open my $MEM_FH, '<', $_[1]; binmode $MEM_FH, ':raw';

    my $total = 0;

    my @indelDbData;
    my @indelRef;
    my $chr;
    my $inputRef;
    my @lines;
    my $dataFromDbAref;
    my $dbReference;
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

      $dataFromDbAref = $self->{_db}->dbReadOne($fields[0], $fields[1] - 1, 1);

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
            $self->{_db}->dbRead($fields[0], \@indelDbData, 1);

            #Note that the first position keeps the same $inputRef
            #This means in the (rare) discordant multiallelic situation, the reference
            #Will be identical between the SNP and DEL alleles
            #faster than perl-style loop (much faster than c-style)
            @indelRef = ($fields[3], map { $self->{_refTrackGetter}->get($_) } @indelDbData);

            #Add the db data that we already have for this position
            unshift @indelDbData, $dataFromDbAref;
          }
        } else {
          #It's an insertion, we always read + 1 to the position being annotated
          # which itself is + 1 from the db position, so we read  $out[1][0][0] to get the + 1 base
          # Read without committing by using 1 as last argument
          @indelDbData = ($dataFromDbAref, $self->{_db}->dbReadOne($fields[0], $fields[1], 1));

          #Note that the first position keeps the same $inputRef
          #This means in the (rare) discordant multiallelic situation, the reference
          #Will be identical between the SNP and DEL alleles
          @indelRef = ($fields[3], $self->{_refTrackGetter}->get($indelDbData[1]));
        }
      }

      if(@indelDbData) {
        ############### Gather all track data (besides reference) #################
        for my $posIdx (0 .. $#indelDbData) {
          for my $track (@trackGettersExceptReference) {
            $fields[$trackIndices->{$track->name}] //= [];

            $track->get($indelDbData[$posIdx], $fields[0], $indelRef[$posIdx], $fields[4], 0, $posIdx, $fields[$trackIndices->{$track->name}]);
          }

          $fields[$refTrackIdx][0][$posIdx] = $indelRef[$posIdx];
        }

        # If we have multiple indel alleles at one position, need to clear stored values
        @indelDbData = ();
        @indelRef = ();
      } else {
        for my $track (@trackGettersExceptReference) {
          $fields[$trackIndices->{$track->name}] //= [];
          $track->get($dataFromDbAref, $fields[0], $fields[3], $fields[4], 0, 0, $fields[$trackIndices->{$track->name}])
        }

        $fields[$refTrackIdx][0][0] = $self->{_refTrackGetter}->get($dataFromDbAref);
      }

       # 3 holds the input reference, we'll replace this with the discordant status
      $fields[3] = $self->{_refTrackGetter}->get($dataFromDbAref) ne $fields[3] ? 1 : 0;
      push @lines, \@fields;
    }

    if(@lines) {
      MCE->gather(scalar @lines, $total - @lines, undef, $self->{_outputter}->makeOutputString(\@lines));
    } else {
      MCE->gather(0, $total);
    }
  } $fh;

  # Force flush
  &$progressFunc(0, 0, undef, undef, 1);

  MCE::Loop::finish();

  # Unfortunately, MCE::Shared::stop() removes the value of $abortErr
  # according to documentation, and I did not see mention of a way
  # to copy the data from a scalar, and don't want to use a hash for this alone
  # So, using a scalar ref to abortErr in the gather function.
  if($abortErr) {
    say "Aborted job";

    # Database & tx need to be closed
    $self->{_db}->cleanUp();

    return ('Job aborted due to error', undef);
  }

  ################ Finished writing file. If statistics, print those ##########
  # Sync to ensure all files written
  close $outFh;

  if($statsFh) {
    close $statsFh;
  }

  system('sync');

  $err = $self->_moveFilesToOutputDir();

  # If we have an error moving the output files, we should still return all data
  # that we can
  if($err) {
    $self->log('error', $err);
  }

  $self->{_db}->cleanUp();

  return ($err || undef, $self->outputFilesInfo);
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

sub _errorWithCleanup {
  my ($self, $msg) = @_;

  $self->log('error', $msg);

  $self->{_db}->cleanUp();

  return $msg;
}

__PACKAGE__->meta->make_immutable;

1;
