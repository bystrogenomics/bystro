
use 5.10.0;
use strict;
use warnings;

package SeqElastic;
use Search::Elasticsearch;

use Path::Tiny;
use Types::Path::Tiny qw/AbsFile AbsPath AbsDir/;
use Mouse 2;
use List::MoreUtils qw/first_index/;

our $VERSION = '0.001';

# ABSTRACT: Index an annotated snpfil

use namespace::autoclean;

use DDP;

use Seq::Output::Delimiters;
use MCE::Loop;
use MCE::Shared;
use Try::Tiny;
use Math::SigFigs qw(:all);
use Scalar::Util qw/looks_like_number/;
use Sys::CpuAffinity;

# An archive, containing an "annotation" file
has annotatedFilePath => (is => 'ro', isa => AbsFile, coerce => 1,
  writer => '_setAnnotatedFilePath');

has indexConfig => (is => 'ro', isa=> 'HashRef', coerce => 1, required => 1);

has connection => (is => 'ro', isa=> 'HashRef', coerce => 1, required => 1);

has publisher => (is => 'ro');

has logPath => (is => 'ro', isa => AbsPath, coerce => 1);

has debug => (is => 'ro');

has verbose => (is => 'ro');

# Probably the user id
has indexName => (is => 'ro', required => 1);

# The index type; probably the job id
has indexType => (is => 'ro', required => 1);

has dryRun => (is => 'ro');

# has commitEvery => (is => 'ro', default => 1000);

# If inputFileNames provided, inputDir is required
has inputDir => (is => 'ro', isa => AbsDir, coerce => 1);

# The user may have given some header fields already
# If so, this is a re-indexing job, and we will want to append the header fields
has headerFields => (is => 'ro', isa => 'ArrayRef');

# The user may have given some additional files
# We accept only an array of bed file here
# TODO: implement
has addedFiles => (is => 'ro', isa => 'ArrayRef');

# IF the user gives us an annotated file path, we will first index from
# The annotation file wihin that archive
#@ params
# <Object> filePaths @params:
  # <String> compressed : the name of the compressed folder holding annotation, stats, etc (only if $self->compress)
  # <String> converted : the name of the converted folder
  # <String> annnotation : the name of the annotation file
  # <String> log : the name of the log file
  # <Object> stats : the { statType => statFileName } object
# Allows us to use all to to extract just the file we're interested from the compressed tarball
has inputFileNames => (is => 'ro', isa => 'HashRef');

has maxThreads => (is => 'ro', isa => 'Int', lazy => 1, default => sub {
  return Sys::CpuAffinity::getNumCpus();
});

with 'Seq::Role::IO', 'Seq::Role::Message', 'MouseX::Getopt';

sub BUILD {
  my $self = shift;

  my $delims = Seq::Output::Delimiters->new();
  # We currently don't support multiallelics on one line
  # $self->{_alleleDelim} = $delims->alleleDelimiter;

  $self->{_fieldSplit} = $delims->splitByField;
  $self->{_overlapSplit}= $delims->splitByOverlap;
  $self->{_posSplit} = $delims->splitByPosition;
  $self->{_valSplit} = $delims->splitByValue;

  $self->{_missChar} = $delims->emptyFieldChar;
  $self->{_overlapDelim} = $delims->overlapDelimiter;

  # Initialize messaging to the queue and logging 
  Seq::Role::Message::initialize();

  # Seq::Role::Message settings
  # We manually set the publisher, logPath, verbosity, and debug, because
  # Seq::Role::Message is meant to be consumed globally, but configured once
  # Treating publisher, logPath, verbose, debug as instance variables
  # would result in having to configure this class in every consuming class
  if(defined $self->publisher) {
    $self->setPublisher($self->publisher);
  }

  if(defined $self->logPath) {
    $self->setLogPath($self->logPath);
  }

  if(defined $self->verbose) {
    $self->setVerbosity($self->verbose);
  }

  #todo: finisih ;for now we have only one level
  if ($self->debug) {
    $self->setLogLevel('DEBUG');
  } else {
    $self->setLogLevel('INFO');
  }

  if(defined $self->inputFileNames) {
    if(!defined $self->inputDir) {
      $self->log('warn', "If inputFileNames provided, inputDir required");
      return ("If inputFileNames provided, inputDir required", undef);
    }

    if(!defined $self->inputFileNames->{archived}
    && !defined $self->inputFileNames->{annnotation}  ) {
      $self->log('warn', "annotation key required in inputFileNames when compressed key has a value");
      return ("annotation key required in inputFileNames when compressed key has a value", undef);
    }
  } elsif(!defined $self->annotatedFilePath) {
    $self->log('warn', "if inputFileNames not provided, annotatedFilePath must be passed");
    return ("if inputFileNames not provided, annotatedFilePath must be passed", undef);
  }
}

sub go {
  my $self = shift;
  
  $self->log( 'info', 'Beginning indexing' );

  my ($filePath, $annotationFileInCompressed) = $self->_getFilePath();

  my ($err, undef, $fh) = $self->getReadFh($filePath, $annotationFileInCompressed);

  if($err) {
    $self->_errorWithCleanup($err);
    return ($err, undef);
  };

  my $firstLine = <$fh>;
  chomp $firstLine;

  $err = $self->_taintCheck($firstLine);

  if($err) {
    $self->_errorWithCleanup($err);
    return ($err, undef);
  }

  my ($headerAref, $pathAref) = $self->_getHeaderPath($firstLine);

  # ES since > 5.2 deprecates lenient boolean
  my @booleanHeaders = _getBooleanHeaders($headerAref, $self->indexConfig->{mappings});

  # We need to flush at the end of each chunk read; so chunk size directly
  # controls bulk request size, as long as bulk request doesnt hit
  # max_count and max_size thresholds
  my $chunkSize = $self->getChunkSize($filePath, $self->maxThreads, 3_000, 10_000);

  # Report every 10k lines; to avoid oversaturating receiver
  my $messageFreq = (1e4 / 4) * $self->maxThreads;

  my $progressFunc = $self->makeLogProgress($messageFreq);

  MCE::Loop::init {
    max_workers => $self->maxThreads, use_slurpio => 1, #Disable on shared storage: parallel_io => 1,
    chunk_size => $chunkSize . 'K',
    gather => $progressFunc,
  };

  my $mutex = MCE::Mutex->new;
  tie my $abortErr, 'MCE::Shared', '';

  # TODO: can use connection pool sniffing as well, not doing so atm
  # because not sure if connection sniffing issue exists here as in
  # elasticjs library
  my $esO = $self->_makeIndex();

  mce_loop_f {
    my ($mce, $slurp_ref, $chunk_id) = @_;

    if($abortErr) {
      say "abort error found";
      $mce->abort();
    }

    my $es = Search::Elasticsearch->new($self->connection);
    my $bulk = $self->_makeBulkHelper($es, \$abortErr, $mutex);

    open my $MEM_FH, '<', $slurp_ref; binmode $MEM_FH, ':raw';

    my $idxCount = 0;
    my $rowDocHref;
    while ( my $line = $MEM_FH->getline() ) {
      # $rowDocHref = $self->_processLine($line, $pathAref);

      $bulk->index({
        source => $self->_processLine($line, $pathAref)
      });

      $idxCount++;
    }

    # Without this, I cannot get all documents to show...
    $bulk->flush();

    MCE->gather($idxCount);
  } $fh;

  # Flush
  $progressFunc->(0, 1);

  MCE::Loop::finish();

  # Disabled for now, we have many abort errors 
  if($abortErr) {
    MCE::Shared::stop();
    say "Error creating index";
    return ("Error creating index", undef, undef);
  }

  # Needed to close some kinds of file handles
  # Doing here for consistency, and the eventuality that we will 
  # modify this with something unexpected

  # Re-enable replicas
  $esO->indices->put_settings(
    index => $self->indexName,
    body => $self->indexConfig->{post_index_settings},
  );

  $esO->indices->refresh(
    index => $self->indexName,
  );

  $self->log('info', "finished indexing");

  return (undef, $headerAref, $self->indexConfig);
}

sub _makeBulkHelper {
  my ($self, $es, $abortErrRef, $mutex) = @_;

  my $bulk = $es->bulk_helper(
    index       => $self->indexName,
    type        => $self->indexType,
    max_count   => 5_000,
    max_size    => 10_000_000,
    on_error    => sub {
      my ($action, $response, $i) = @_;
      $self->log('warn', "Index error: $action ; $response ; $i");
      p @_;
      $mutex->synchronize(sub{ $$abortErrRef = $response} );
    },           # optional
    on_conflict => sub {
      my ($action,$response,$i,$version) = @_;
      $self->log('warn', "Index conflict: $action ; $response ; $i ; $version");
    },           # optional
  );

  return $bulk
}

sub _makeIndex {
  my $self = shift;

  my $es = Search::Elasticsearch->new($self->connection);

  if(!$es->indices->exists(index => $self->indexName) ) {
    $es->indices->create(index => $self->indexName, body => {settings => $self->indexConfig->{index_settings}});
  } else {
    # Index must be open to put index settings
    $es->indices->open(index => $self->indexName);

    # Will result in errors [illegal_argument_exception] can't change the number of shards for an index
    # $es->indices->put_settings(
    #   index => $self->indexName,
    #   body => $searchConfig->{index_settings},
    # );
  }

  $es->indices->put_mapping(
    index => $self->indexName,
    type => $self->indexType,
    body => $self->indexConfig->{mappings},
  );

  return $es;
}

sub _getHeaderPath {
  my ($self, $line) = @_;
  chomp $line;

  my @header = $self->{_fieldSplit}->($line);

  my @paths = @header;
  for (my $i = 0; $i < @paths; $i++) {
    if( index($paths[$i], '.') > -1 ) {
      $paths[$i] = [ split(/[.]/, $paths[$i]) ];
    }
  }

  return (\@header, \@paths);
}

sub _taintCheck {
  my ($self, $line) = @_;

  my $taintCheckRegex = $self->taintCheckRegex;

  if ( $line !~ /$taintCheckRegex/xgm ) {
    return "First line of input file has illegal characters";
  }

  return;
}

sub _processLine {
  my ($self, $line, $pathsAref) = @_;

  chomp $line;

  my %rowDocument;
  my $i = -1;
  my $od = $self->{_overlapDelim};
  my $miss = $self->{_missChar};
  for my $field ($self->{_fieldSplit}->($line)) {
    $i++;

    if($field eq $self->{_missChar}) {
      next;
    }

    # say STDERR "Field: $field ; i : $i; path: " . ( ref $pathsAref->[$i] ? join(".", @{$pathsAref->[$i]}) : $pathsAref->[$i] );
    # Every value is stored @ [alleleIdx][positionIdx]
    my @posVals;

    # TODO: If or when we introduce alleleDelimiter split we will need to remove
    # [ @values > 1 ? \@values : $values[0] ] and replace with
    # @values > 1 ? \@values : $values[0]

      POS_LOOP: for my $posValue ($self->{_posSplit}->($field)) {
        if ($posValue eq $miss) {
          push @posVals, undef;

          next;
        }

        my @values;
        if(index($posValue, $od) == -1) {
          @values = map { $_ ne $miss ? $_ : undef } $self->{_valSplit}->($posValue);

          push @posVals, @values > 1 ? \@values : $values[0];
          next;
        }

        for my $val ($self->{_valSplit}->($posValue)) {
          my @inner = map { $_ ne $miss ? $_ : undef } $self->{_overlapSplit}->($val);

          push @values, @inner > 1 ? \@inner : $inner[0];
        }

        push @posVals, @values > 1 ? \@values : $values[0];
      }

      # The brackets around \@posVals are to set hierarchy for
      # in the future allowing merger of multiple alleles (rows)
      # under that field (as in multiallelics to show independent effects)
      _populateHashPath(\%rowDocument, $pathsAref->[$i], [\@posVals]);
  }

  return \%rowDocument;
}

sub makeLogProgress {
  my $self = shift;
  my $throttleThreshold = shift;

  my $total = 0;

  my $hasPublisher = $self->hasPublisher;

  if(!$hasPublisher) {
    # noop
    return sub{};
  }

  if(!$throttleThreshold) {
    $throttleThreshold = 1e4;
  }

  my $throttleIndicator = 0;
  return sub {
    #my $progress, $flush = shift;
    ##    $_[0]  , $_[1]

    # send messages only every $throttleThreshold, to avoid overloading publishing server
    if(defined $_[0]) {
      $throttleIndicator += $_[0];

      if($_[1] || $throttleIndicator >= $throttleThreshold) {
        $total += $throttleIndicator;

        $self->publishProgress($total);

        $throttleIndicator = 0;
      }
    }
  }
}

sub _populateHashPath {
  my ($hashRef, $pathAref, $dataForEndOfPath) = @_;
  #     $_[0]  , $_[1]    , $_[2]

  if(!ref $pathAref) {
    $hashRef->{$pathAref} = $dataForEndOfPath;
    return $hashRef;
  }

  my $href = $hashRef;
  for (my $i = 0; $i < @$pathAref; $i++) {
    if($i + 1 == @$pathAref) {
      $href->{$pathAref->[$i]} = $dataForEndOfPath;
    } else {
      if(!defined  $href->{$pathAref->[$i]} ) {
        $href->{$pathAref->[$i]} = {};
      }

      $href = $href->{$pathAref->[$i]};
    }
  }

  return $href;
}

sub _getFilePath {
  my $self = shift;

  if($self->inputFileNames && $self->inputDir) {
    # The user wants us to make the annotation_file_path
    if(defined $self->inputFileNames->{archived}) {
      # The user had compressed this file (see Seq.pm)
      # This is expected to be a tarball, which we will extract, but only
      # to stream the annotation file within the tarball package
      my $path = $self->inputDir->child($self->inputFileNames->{archived});

      return ($path, $self->inputFileNames->{annotation})
    }

    if($self->debug) {
      say "in _getFilePath inputFileNames";
      p $self->inputFileNames;
      p $self->inputDir;
    }

    my $path = $self->inputDir->child($self->inputFileNames->{annotation});

    return ($path, undef);
  }

  return $self->annotatedFilePath;
}

sub _getBooleanMappings {
  my ($mapRef, $parentName) = @_;

  my @booleanMappings;

  my $pVal;
  for my $property (keys %{$mapRef->{properties}}) {
    $pVal = $mapRef->{properties}{$property};

    if(exists $pVal->{fields}) {
      for my $subProp (keys %{$pVal->{fields}}) {
        if(exists $pVal->{fields}{$subProp}{type} && $pVal->{fields}{$subProp}{type} eq 'boolean') {
          push @booleanMappings, "$property.$subProp";
        }
      }
    } elsif(exists $pVal->{properties}) {
      push @booleanMappings, _getBooleanMappings($pVal, $property);
    } elsif(exists $pVal->{type} && $pVal->{type} eq 'boolean') {
      push @booleanMappings, $property;
    }
  }

  return @booleanMappings;
}

sub _getBooleanHeaders {
  my ($headerAref, $mapping) = @_;

  my %booleanValues = map { $_ => 1 } _getBooleanMappings($mapping);

  my @booleanHeaders = map { $booleanValues{$_} ? 1 : 0 } @$headerAref;

  return @booleanHeaders;
}


sub _errorWithCleanup {
  my ($self, $msg) = @_;

  # To send a message to clean up files.
  # TODO: Need somethign better
  #MCE->gather(undef, undef, $msg);

  $self->log('warn', $msg);
  return $msg;
}

__PACKAGE__->meta->make_immutable;

1;
