use 5.10.0;
use strict;
use warnings;

# ABSTRACT: Create an annotation from a query
# TODO: maxThreads is ignored.
# 1) igure out if we want to change that.
#    - It is ignored because the elasticsearch cluster is too slow on a per-shard basis
# 2) Document it

package SeqFromQuery;
our $VERSION = '0.001';

use namespace::autoclean;
use DDP;
use lib './lib';
use Mouse 2;

use MCE::Loop;

use Search::Elasticsearch;

use Seq::Output::Delimiters;

use Cpanel::JSON::XS qw/decode_json encode_json/;
use YAML::XS qw/LoadFile/;

use Math::Round qw/nhimult/;

# Defines basic things needed in builder and annotator, like logPath,
# Also initializes the database with database_dir
# TODO: could move away from initializing the LMDB database, not needed here
extends 'Seq::Base';

# Defines most of the properties that can be configured at run time
# Needed because there are variations of Seq.pm, ilke SeqFromQuery.pm
with 'Seq::Definition';

# An archive, containing an "annotation" file
has inputQueryBody => (is => 'ro', isa => 'HashRef', required => 1);

# Probably the user id
has indexName => (is => 'ro', required => 1);

# The index type; probably the job id
has indexType => (is => 'ro', required => 1);

has assembly => (is => 'ro', isa => 'Str', required => 1);

has configPath => (is => 'ro', isa => 'Str', default => 'config/');

# has commitEvery => (is => 'ro', default => 5000);

# The user may have given some header fields already
# If so, this is a re-indexing job, and we will want to append the header fields
has fieldNames => (is => 'ro', isa => 'ArrayRef', required => 1);

has indexConfig => (is => 'ro', isa => 'HashRef', required => 1);

# Elasticsearch connection configuration
has connection => (is => 'ro', isa => 'HashRef', required => 1);

has batchSize => (is => 'ro', isa => 'Num', default => 5000);

has shards => (is => 'ro', isa => 'Num', lazy => 1, default => sub {
  my $self = shift;

  return $self->indexConfig->{index_settings}->{index}->{number_of_shards};
});

has maxShards => (is => 'ro', isa => 'Num');

my $prettyCoder = Cpanel::JSON::XS->new->ascii->pretty->allow_nonref;;
# TODO: This is too complicated, shared with Seq.pm for the most part

around BUILDARGS => sub {
  my ($orig, $class, $href) = @_;

  my %data = %$href;

  if($data{indexConfig}) {
    if(!ref $data{indexConfig}) {
      $data{indexConfig} = decode_json($data{indexConfig});
    }

    return $class->$orig(\%data);
  }

  my $cf = path('config')->child($data{assembly} . '.mapping.yml')->stringify();

  $data{indexConfig} = LoadFile($cf);

  return $class->$orig(\%data);
};

sub BUILD {
  my $self = shift;

  # Makes or fails silently if exists
  $self->outDir->mkpath();

  if(!$self->shards) {
    $self->log('fatal', "Cannot read number of shards in index");
  }
}

sub annotate {
  my $self = shift;

  $self->log( 'info', 'Beginning saving annotation from query' );

  $self->log( 'info', 'Input query is: ' . $prettyCoder->encode($self->inputQueryBody) );

  my $d = Seq::Output::Delimiters->new();

  my %delims = (
    'allele' => $d->alleleDelimiter,
    'pos' => $d->positionDelimiter,
    'value' => $d->valueDelimiter,
    'overlap' => $d->overlapDelimiter,
    'miss' => $d->emptyFieldChar,
    'fieldSep' => $d->fieldSeparator,
  );

  ################## Make the full output path ######################
  # The output path always respects the $self->output_file_base attribute path;
  my ($err, $outFh, $statsFh) = $self->_getFileHandles();

  if ($err) {
    $self->_errorWithCleanup($err);
    return ($err, undef);
  }

  my $hasSort = $self->inputQueryBody->{sort} && $self->inputQueryBody->{sort} ne '_id';

  my ($parentsAref, $childrenAref) = $self->_getHeader();

  my @childrenOrOnly = @$childrenAref;
  my @parentNames = @$parentsAref;

  my @fieldNames = @{$self->fieldNames};
  my $outputHeader = join($delims{fieldSep}, @fieldNames);

  say $outFh $outputHeader;

  if($statsFh) {
    say $statsFh $outputHeader;
  }

  $self->log('info', "Beginning to create annotation from the query");

  ($err, my $discordantIdx) = $self->_getDiscordantIdx();

  if($err) {
    $self->log('fatal', "Couldn't find discordant index");
  }

  my $nearestMultiple;

  my $nSlices = $self->_getSlices();

  my $slice = {
    field => 'pos',
    max => $nSlices,
  };

  my $progressFunc = $self->_makeLogProgress($hasSort, $outFh, $statsFh, 3e4);

  MCE::Loop::init {
    max_workers => $nSlices,
    chunk_size => 1,
    gather => $progressFunc,
  };

  # We do parallel fetch using sliced scroll queries.
  # It is far too expensive to do a single-threaded scroll
  # and pass the entire structure to MCE;
  # Will get "recursion limit" errors in Storable due to the size of the structure
  # for bulk queries
  mce_loop {
    my ($mce, $chunkRef, $chunkId) = @_;

    my $id = $_;

    my $es = Search::Elasticsearch->new($self->connection);

    $slice->{id} = $id;
    $self->inputQueryBody->{slice} = $slice;

    my $scroll = $es->scroll_helper(
      size        => $self->batchSize,
      body        => $self->inputQueryBody,
      index => $self->indexName,
      type => $self->indexType,
    );

    while(my @docs = $scroll->next($self->batchSize)) {
      my @sourceData;
      $#sourceData = $#docs;

      my $i = 0;
      for my $doc (@docs) {
        my @rowData;
        # Initialize all values to undef
        # Output.pm requires a sparse array for any missing values
        # To preserve output order
        $#rowData = $#fieldNames;

        for my $y (0 .. $#fieldNames) {
          $rowData[$y] = _populateArrayPathFromHash($childrenOrOnly[$y], $doc->{_source}{$parentNames[$y]});
        }

        if($rowData[$discordantIdx][0][0] eq 'false') {
          $rowData[$discordantIdx][0][0] = 0;
        } elsif($rowData[$discordantIdx][0][0] eq 'true') {
          $rowData[$discordantIdx][0][0] = 1;
        }

        $sourceData[$i] = \@rowData;

        $i++;
      }

      my $outputString = _makeOutputString(\@sourceData, \%delims);

      $mce->gather(scalar @docs, $outputString, $id);
    }
  } (0 .. $nSlices - 1);

  # Flush
  $progressFunc->(0, undef, undef, 1);

  MCE::Loop::finish();

  ################ Finished writing file. If statistics, print those ##########
  # First sync output to ensure everything needed is written
  # then close all file handles and move files to output dir
  $err = $self->safeSystem('sync')
         ||
         $self->safeClose($outFh)
         ||
         ($statsFh && $self->safeClose($statsFh))
         ||
         $self->_moveFilesToOutputDir();

  if($err) {
    $self->_errorWithCleanup($err);

    return ($err, undef);
  }

  return ($err, $self->outputFilesInfo);
}

sub _getSlices {
  my $self = shift;

  my $nShards = $self->shards;
  my $nThreads = $self->maxThreads;

  if($nShards < $nThreads) {
    my $divisor = nhimult(2,$nThreads / $nShards);

    return $nShards * $divisor ;
  }

  # each thread runs at < 100% utilization
  return $nShards * 2;
}

sub _getHeader {
  my $self = shift;

  my @fieldNames = @{$self->fieldNames};;

  my @childrenOrOnly;
  $#childrenOrOnly = $#fieldNames;

  # Elastic top level of { parent => child } is parent.
  my @parentNames;
  $#parentNames = $#fieldNames;

  for my $i (0 .. $#fieldNames) {
    if( index($fieldNames[$i], '.') > -1 ) {
      my @path = split(/\./, $fieldNames[$i]);
      $parentNames[$i] = $path[0];

      if(@path == 2) {
        $childrenOrOnly[$i] = [ $path[1] ];
      } elsif(@path > 2) {
        $childrenOrOnly[$i] = [ @path[ 1 .. $#path] ];
      }

    } else {
      $parentNames[$i] = $fieldNames[$i];
      $childrenOrOnly[$i] = $fieldNames[$i];
    }
  }

  return (\@parentNames, \@childrenOrOnly);
}

sub _populateArrayPathFromHash {
  my ($pathAref, $dataForEndOfPath) = @_;
  #     $_[0]  , $_[1]    , $_[2]
  if(!ref $pathAref) {
    return $dataForEndOfPath;
  }

  for my $i (0 .. $#$pathAref) {
    $dataForEndOfPath = $dataForEndOfPath->{$pathAref->[$i]};
  }

  return $dataForEndOfPath;
}

sub _makeOutputString {
  my ($arrayRef, $delims) = @_;

  my $emptyFieldChar = $delims->{miss};
  # Expects an array of row arrays, which contain an for each column, or an undefined value
  for my $row (@$arrayRef) {
    COLUMN_LOOP: for my $column (@$row) {
      # Some fields may just be missing; we won't store even the
      # alt/pos [[]] structure for those
      if(!defined $column) {
        $column = $emptyFieldChar;
        next COLUMN_LOOP;
      }

      for my $alleleData (@$column) {
        POS_LOOP: for my $positionData (@$alleleData) {
          if(!defined $positionData) {
            $positionData = $emptyFieldChar;
            next POS_LOOP;
          }

          if(ref $positionData) {
            $positionData = join($delims->{value}, map {
              defined $_
              ?
              (ref $_ ? join($delims->{overlap}, @$_) : $_)
              : $emptyFieldChar
            } @$positionData);
            next POS_LOOP;
          }
        }

        $alleleData = join($delims->{pos}, @$alleleData);
      }

      $column = join($delims->{allele}, @$column);
    }

    $row = join($delims->{fieldSep}, @$row);
  }

  return join("\n", @$arrayRef);
}

sub _makeLogProgress {
  my ($self, $hasSort, $outFh, $statsFh, $throttleThreshold) = @_;

  if(!$throttleThreshold) {
    $throttleThreshold = 2e4;
  }

  my $throttleIndicator = 0;

  my $total = 0;

  my $hasPublisher = $self->hasPublisher;

  # my %result;

  my $orderId = 0;
  return sub {
    my ($progress, $outputStringRef, $chunkId, $flush) = @_;

    $throttleIndicator += $progress;

    if($hasPublisher) {
      $total += $progress;

      if($throttleIndicator >= $throttleThreshold || $flush) {
        $self->publishProgress($total);
        $throttleIndicator = 0;
      }
    }

    if(!$outputStringRef) {
      return;
    }

    say $statsFh $outputStringRef;
    say $outFh $outputStringRef;


    # TODO: Make ordered print work
  #   while (1) {
  #     if(!exists $result{$orderId}) {
  #       $orderId = 0;
  #       last;
  #     }

  #     say $outFh $result{$orderId};

  #     delete $result{$orderId};

  #     $orderId++;

  #   }

    return;
  }
}

sub _getFileHandles {
  my $self = shift;

  my ($err, $outFh, $statsFh);

  # _working dir from Seq::Definition
  my $outPath = $self->_workingDir->child($self->outputFilesInfo->{annotation});

  ($err, $outFh) = $self->getWriteFh($outPath);

  if($err) {
    return ($err, undef, undef);
  }

  if(!$self->run_statistics) {
    return ($err, $outFh, $statsFh);
  }

  my $args = $self->_statisticsRunner->getStatsArguments();
  $err = $self->safeOpen($statsFh, "|-", $args);

  if($err) {
    return ($err, undef, undef);
  }

  return ($err, $outFh, $statsFh);
}

sub _getDiscordantIdx {
  my $self = shift;

  my $idx = 0;
  my $didx = -1;

  for my $field (@{$self->fieldNames}) {
    if ($field eq 'discordant') {
      $didx = $idx;
      last;
    }

    $idx++;
  }

  if($didx == -1) {
    return ("Couldn't find index", undef);
  }

  return (undef, $didx);
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
