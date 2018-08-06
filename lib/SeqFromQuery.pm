use 5.10.0;
use strict;
use warnings;

# ABSTRACT: Create an annotation from a query
package SeqFromQuery;
use lib './lib';
use Mouse 2;
our $VERSION = '0.001';
use MCE::Loop;
use Search::Elasticsearch;
use DDP;

use namespace::autoclean;

use Seq::Output::Delimiters;

use Cpanel::JSON::XS qw/decode_json encode_json/;

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
# has commitEvery => (is => 'ro', default => 5000);

# The user may have given some header fields already
# If so, this is a re-indexing job, and we will want to append the header fields
has fieldNames => (is => 'ro', isa => 'ArrayRef', required => 1);

# Elasticsearch connection configuration
has connection => (is => 'ro', isa => 'HashRef', required => 1);

my $prettyCoder = Cpanel::JSON::XS->new->ascii->pretty->allow_nonref;;
# TODO: This is too complicated, shared with Seq.pm for the most part
sub BUILD {
  my $self = shift;

  # Makes or fails silently if exists
  $self->outDir->mkpath();
}

sub annotate {
  my $self = shift;

  $self->log( 'info', 'Beginning saving annotation from query' );

  $self->log( 'info', 'Input query is: ' . $prettyCoder->encode($self->inputQueryBody) );

  my $hasSort = $self->inputQueryBody->{sort};

  my $delims = Seq::Output::Delimiters->new();

  my $overlapDelim = $delims->overlapDelimiter;
  my $posDelim = $delims->positionDelimiter;
  my $valueDelim = $delims->valueDelimiter;
  my $fieldSep = $delims->fieldSeparator;
  my $missChar = $delims->emptyFieldChar;

  my ($parentsAref, $childrenAref) = $self->_getHeader();
  my @childrenOrOnly = @$childrenAref;
  my @parentNames = @$parentsAref;

  ################## Make the full output path ######################
  # The output path always respects the $self->output_file_base attribute path;
  my $outputPath = $self->_workingDir->child($self->outputFilesInfo->{annotation});

  my ($err, undef, $outFh) = $self->getWriteFh($outputPath);

  if(!$err) {
    $self->_errorWithCleanup($err);
    return ($err, undef);
  }

  # Stats may or may not be provided
  my @fieldNames = @{$self->fieldNames};
  my $outputHeader = join($fieldSep, @fieldNames);

  say $outFh $outputHeader;

  my $statsFh;
  if($self->run_statistics) {
    my $args = $self->_statisticsRunner->getStatsArguments();
    $err = $self->safeOpen($statsFh, "|-", $args);

    if($err) {
      $self->_errorWithCleanup($err);
      return ($err, undef);
    }

    say $statsFh $outputHeader;
  }

  $self->log('info', "Beginning to create annotation from the query");

  my $batchSize = 4000;

  MCE::Loop::init {
    max_workers => $self->maxThreads || 1, chunk_size => $batchSize,
    gather => $self->_makeLogProgress($hasSort, $statsFh, $outFh)
  };

  mce_loop {
    my ($mce, $chunkRef, $chunkId) = @_;

    my @sourceData;
    $#sourceData = $#$_;
    my $i = 0;

    for my $doc (@$_) {
      my @rowData;
      # Initialize all values to undef
      # Output.pm requires a sparse array for any missing values
      # To preserve output order
      $#rowData = $#fieldNames;

      for my $y (0 .. $#fieldNames) {
        $rowData[$y] = _populateArrayPathFromHash($childrenOrOnly[$y], $doc->{_source}{$parentNames[$y]});
      }

      $sourceData[$i] = \@rowData;

      $i++;
    }

    my $outputString = _makeOutputString(\@sourceData, 
      $missChar, $valueDelim, $posDelim, $overlapDelim, $fieldSep);

    $mce->gather(scalar @$_, $outputString, $chunkId);
  } $self->_esIterator($batchSize);

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

sub _esIterator {
  my ($self, $batchSize) = @_;
  # TODO: can use connection pool sniffing as well, not doing so atm
  # because not sure if connection sniffing issue exists here as in
  # elasticjs library
  my $es = Search::Elasticsearch->new($self->connection);

  my $scroll = $es->scroll_helper(
    size        => $batchSize,
    body        => $self->inputQueryBody,
    index => $self->indexName,
    type => $self->indexType,
  );

  return sub {
     my ($chunkSize) = @_;

     my @docs = $scroll->next($chunkSize);
     if (@docs) {
        return @docs;
     }

     return;
  };
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
  my ($arrayRef, $missChar, $valueDelim, $posDelim, $overlapDelim, $fieldSep) = @_;

  # Expects an array of row arrays, which contain an for each column, or an undefined value
  for my $row (@$arrayRef) {
    # Columns are features
    COLUMN_LOOP: for my $column (@$row) {
      # Some fields may just be missing
      if(!defined $column) {
        $column = $missChar;
        next COLUMN_LOOP;
      }

      # Most sites have a single position (not indels)
      # Avoid a loop if so
      if(@$column == 1) {
        if(ref $column->[0]) {
          $column = join($valueDelim, map {
            !defined $_
            ?
            $missChar
            :
            ( ref $_
              ?
              join($overlapDelim, map { defined $_ ? $_ : $missChar } @$_)
              :
              $_
            )
          } @{$column->[0]});

          next COLUMN_LOOP;
        }

        $column = defined $column->[0] ? $column->[0] : $missChar;

        next COLUMN_LOOP;
      }

      POS_LOOP: for my $positionData (@$column) {
        if(!defined $positionData) {
          $positionData = $missChar;
          next POS_LOOP;
        }

        if(ref $positionData) {
          $positionData = join($valueDelim, map {
            !defined $_
            ?
            $missChar
            :
            ( ref $_
              ?
              join($overlapDelim, map { defined $_ ? $_ : $missChar } @$_)
              :
              $_
            )
          } @$positionData);

          next POS_LOOP;
        }
      }

      $column = join($posDelim, @$column);
    }

    $row = join($fieldSep, @$row);
  }

  return join("\n", @$arrayRef);
}

sub _makeLogProgress {
  my ($self, $hasSort, $outFh, $statsFh) = @_;

  my $total = 0;

  my $hasPublisher = $self->hasPublisher;

  my %result;
  my $orderId = 1;

  if($hasSort) {
    return sub {
      my ($progress, $outputStringRef, $chunkId) = @_;

      if($hasPublisher) {
        $total += $progress;
        $self->publishProgress($total);
      }

      $result{ $chunkId } = $outputStringRef;

      say $statsFh $outputStringRef;

      while (1) {
        last unless exists $result{$orderId};

        say $outFh $outputStringRef;

        $orderId++;
      }
    }
  }

  return sub {
    my ($progress, $outputStringRef) = @_;

    if($hasPublisher) {
      $total += $progress;
      $self->publishProgress($total);
    }

    say $statsFh $outputStringRef;
    say $outFh $outputStringRef;
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
