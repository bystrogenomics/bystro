use 5.10.0;
use strict;
use warnings;

# ABSTRACT: Create an annotation from a query
package SeqFromQuery;

use Mouse 2;
our $VERSION = '0.001';

use Search::Elasticsearch;

use namespace::autoclean;

use Seq::Output;

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

  my $outputter = Seq::Output->new();

  my $alleleDelimiter = $outputter->delimiters->alleleDelimiter;
  my $positionDelimiter = $outputter->delimiters->positionDelimiter;
  my $valueDelimiter = $outputter->delimiters->valueDelimiter;
  my $fieldSeparator = $outputter->delimiters->fieldSeparator;
  my $emptyFieldChar = $outputter->delimiters->emptyFieldChar;

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

  ################## Make the full output path ######################
  # The output path always respects the $self->output_file_base attribute path;
  my $outputPath = $self->_workingDir->child($self->outputFilesInfo->{annotation});

  my $outFh = $self->get_write_fh($outputPath);

  if(!$outFh) {
    #TODO: should we report $err? less informative, but sometimes $! reports bull
    #i.e inappropriate ioctl when actually no issue
    my $err = "Failed to open " . $self->_workingDir;
    $self->_errorWithCleanup($err);
    return ($err, undef);
  }

  # TODO: can use connection pool sniffing as well, not doing so atm
  # because not sure if connection sniffing issue exists here as in
  # elasticjs library
  my $es = Search::Elasticsearch->new($self->connection);

  my $batchSize = 4000;

  my $scroll = $es->scroll_helper(
    size        => $batchSize,
    body        => $self->inputQueryBody,
    index => $self->indexName,
    type => $self->indexType,
  );

  # Stats may or may not be provided
  my $statsFh;
  my $outputHeader = join($fieldSeparator, @fieldNames);

  # Write the header

  say $outFh $outputHeader;

  # TODO: error handling if fh fails to open
  if($self->run_statistics) {
    my $args = $self->_statisticsRunner->getStatsArguments();
    open($statsFh, "|-", $args) or $self->log('fatal', "Couldnt open Bystro Statistics due to $! (exit code: $?)");

    say $statsFh $outputHeader;
  }

  $self->log('info', "Beginning to create annotation from the query");

  my $progressHandler = $self->makeLogProgress();

  while(my @docs = $scroll->next($batchSize)) {
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

      $sourceData[$i] = \@rowData;

      $i++;
    }

    my $outputString = _makeOutputString(\@sourceData, 
      $emptyFieldChar, $valueDelimiter, $positionDelimiter, $alleleDelimiter, $fieldSeparator);

    print $outFh $outputString . "\n";
    print $statsFh $outputString . "\n";

    &{$progressHandler}(scalar @docs);
  }

  ################ Finished writing file. If statistics, print those ##########
  # Sync to ensure all files written
  close $outFh;

  if($statsFh) {
    close $statsFh;
  }

  system('sync');

  my $err = $self->_moveFilesToOutputDir();

  # If we have an error moving the output files, we should still return all data
  # that we can
  if($err) {
    $self->log('error', $err);
  }

  return ($err || undef, $self->outputFilesInfo);
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
  my ($arrayRef, $emptyFieldChar, $valueDelimiter, $positionDelimiter, $alleleDelimiter, $fieldSeparator) = @_;

  # Expects an array of row arrays, which contain an for each column, or an undefined value
  for my $row (@$arrayRef) {
    COLUMN_LOOP: for my $column (@$row) {
      # Some fields may just be missing
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
            $positionData = join($valueDelimiter, map { defined $_ ? $_ : $emptyFieldChar } @$positionData);
            next POS_LOOP;
          }
        }

        $alleleData = join($positionDelimiter, @$alleleData);
      }

      $column = join($alleleDelimiter, @$column);
    }

    $row = join($fieldSeparator, @$row);
  }

  return join("\n", @$arrayRef);
}

sub makeLogProgress {
  my $self = shift;

  my $total = 0;

  my $hasPublisher = $self->hasPublisher;

  if(!$hasPublisher) {
    # noop
    return sub{};
  }

  return sub {
    #my $progress = shift;
    ##    $_[0] 

    if(defined $_[0]) {

      $total += $_[0];

      $self->publishProgress($total);
      return;
    }
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
