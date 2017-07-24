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

has max_threads => (is => 'ro', isa => 'Int', default => 8);

with 'Seq::Role::IO', 'Seq::Role::Message', 'MouseX::Getopt';

sub getBooleanMappings {
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
      push @booleanMappings, getBooleanMappings($pVal, $property);
    } elsif(exists $pVal->{type} && $pVal->{type} eq 'boolean') {
      push @booleanMappings, $property;
    }
  }

  return @booleanMappings;
}

sub getBooleanHeaders {
  my ($headerAref, $mapping) = @_;

  my %booleanValues = map { $_ => 1 } getBooleanMappings($mapping);

  my @booleanHeaders = map { $booleanValues{$_} ? 1 : 0 } @$headerAref;

  return @booleanHeaders;
}

sub go {
  my $self = shift; $self->log( 'info', 'Beginning indexing' );

  my ($filePath, $annotationFileInCompressed) = $self->_getFilePath();

  (my $err, undef, my $fh) = $self->get_read_fh($filePath, $annotationFileInCompressed);

  if($err) {
    #TODO: should we report $err? less informative, but sometimes $! reports bull
    #i.e inappropriate ioctl when actually no issue
    $self->_errorWithCleanup($!);
    return ($!, undef);
  }

  my $fieldSeparator = $self->delimiter;

  my $taint_check_regex = $self->taint_check_regex; 

  my $firstLine = <$fh>;

  chomp $firstLine;

  my @headerFields;
  if ( $firstLine =~ m/$taint_check_regex/xm ) {
    @headerFields = split $fieldSeparator, $1;
  } else {
    return ("First line of input file has illegal characters", undef);
  }

  my @paths = @headerFields;
  for (my $i = 0; $i < @paths; $i++) {
    if( index($paths[$i], '.') > -1 ) {
      $paths[$i] = [ split(/\./, $paths[$i]) ];
    }
  }

  # ES since > 5.2 deprecates lenient boolean
  my @booleanHeaders = getBooleanHeaders(\@headerFields, $self->indexConfig->{mappings});

  if($self->debug) {
    say "boolean headers are";
    p @booleanHeaders;
  }

  my $delimiters = Seq::Output::Delimiters->new();

  my $alleleDelimiter = $delimiters->alleleDelimiter;
  my $positionDelimiter = $delimiters->positionDelimiter;
  my $valueDelimiter = $delimiters->valueDelimiter;

  my $emptyFieldChar = $delimiters->emptyFieldChar;

  # Todo implement tain check; default taint check doesn't work for annotated files
  # $taint_check_regex = qr/./;
  # With a too-low value, too many messages sent to the receiver
  # slows down apparent indexing
  my $chunkSize = $self->getChunkSize($filePath, $self->max_threads);
  if($chunkSize < 2000) {
    $chunkSize = 2000;
  }

  if($chunkSize > 10000) {
    $chunkSize = 10000;
  }

  # Report every 10k lines; to avoid oversaturating receiver
  my $progressFunc = $self->makeLogProgress(1e4);

  MCE::Loop::init {
    max_workers => 8, use_slurpio => 1, #Disable on shared storage: parallel_io => 1,
    chunk_size => $chunkSize . 'K',
    gather => $progressFunc,
  };

  # TODO: can use connection pool sniffing as well, not doing so atm
  # because not sure if connection sniffing issue exists here as in
  # elasticjs library
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

  my $m1 = MCE::Mutex->new;
  tie my $abortErr, 'MCE::Shared', '';

  my $bulk = $es->bulk_helper(
    index       => $self->indexName,
    type        => $self->indexType,
    on_error    => sub {
      my ($action, $response, $i) = @_;
      $self->log('warn', "Index error: $action ; $response ; $i");
      p @_;
      $m1->synchronize(sub{ $abortErr = $response} );
    },           # optional
    on_conflict => sub {
      my ($action,$response,$i,$version) = @_;
      $self->log('warn', "Index conflict: $action ; $response ; $i ; $version");
    },           # optional
  );

  mce_loop_f {
    my ($mce, $slurp_ref, $chunk_id) = @_;

    my @lines;

    if($abortErr) {
      say "abort error found";
      $mce->abort();
    }

    open my $MEM_FH, '<', $slurp_ref; binmode $MEM_FH, ':raw';

    my $idxCount = 0;
    while ( my $line = $MEM_FH->getline() ) {
      #TODO: implement; the default taint check regex doesn't work for annotated files
      # if (! $line =~ /$taint_check_regex/) {
      #   next;
      # }
      chomp $line;

      my @fields = split $fieldSeparator, $line;

      my %rowDocument;
      my $colIdx = 0;
      my $foundWeird = 0;
      my $alleleIdx;
      my $posIdx;
      my $isBool;

      # We use Perl's in-place modification / reference of looped-over variables
      # http://ideone.com/HNgMf7
      OUTER: for (my $i = 0; $i < @fields; $i++) {
        my $field = $fields[$i];

        #Every value is stored @ [alleleIdx][positionIdx]
        my @out;

        if($field ne $emptyFieldChar) {
          # ES since > 5.2 deprecates lenient boolean
          $alleleIdx = 0;
          ALLELE_LOOP: for my $fieldValue ( split("\\$alleleDelimiter", $field) ) {
            if ($fieldValue eq $emptyFieldChar) {
              # If the value is empty, then this value is for the default/0 index position
              # Means it's not an indel
              $out[$alleleIdx][0] = undef;

              $alleleIdx++;
              next ALLELE_LOOP;
            }

            $posIdx = 0;
            POS_LOOP: for my $posValue ( split("\\$positionDelimiter", $fieldValue) ) {
              if ($posValue eq $emptyFieldChar) {
                $out[$alleleIdx][$posIdx] = undef;

                $posIdx++;
                next POS_LOOP;
              }

              # ES since > 5.2 deprecates lenient boolean
              my @values = map {
                $_ ne $emptyFieldChar ? ($booleanHeaders[$i] ? ($_ ? 'true': 'false') : $_) : undef
              } split("\\$valueDelimiter", $posValue);

              if(!@values) {
                $out[$alleleIdx][$posIdx] = undef;
              } else {
                $out[$alleleIdx][$posIdx] = @values > 1 ? \@values : $values[0];
              }

              $posIdx++;
            }

            $alleleIdx++;
          }

          _populateHashPath(\%rowDocument, $paths[$i], \@out);
        }
      }

      $bulk->index({
        index => $self->indexName,
        type => $self->indexType,
        source => \%rowDocument
      });

      $idxCount++;
    }

    # Without this, I cannot get all documents to show...
    $bulk->flush();

    MCE->gather($idxCount);
  } $fh;

  # Flush
  &$progressFunc(0, 1);

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
  $es->indices->put_settings(
    index => $self->indexName,
    body => $self->indexConfig->{post_index_settings},
  );

  $es->indices->refresh(
    index => $self->indexName,
  );

  $self->log('info', "finished indexing");

  return (undef, \@headerFields, $self->indexConfig);
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

sub BUILD {
  my $self = shift;

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
