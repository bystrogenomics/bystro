use 5.10.0;
use strict;
use warnings;

# ABSTRACT: Create an annotation from a query
# TODO: 1) Support sort
####### 2) Don't initialize db
####### 3) figure out why even with field => 'pos' we get very slow perf

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
use Statistics::Distributions qw(chisqrdistr udistr);

use Math::Round qw/nhimult round/;

use POSIX qw/lround/;

# Defines basic things needed in builder and annotator, like logPath,
# Also initializes the database with database_dir unnecessarily
extends 'Seq::Base';

# Defines most of the properties that can be configured at run time
# Needed because there are variations of Seq.pm, ilke SeqFromQuery.pm
with 'Seq::Definition';

# An archive, containing an "annotation" file
has inputQueryBody => (is => 'ro', isa => 'HashRef', required => 1);

# Post-processing to run, before commiting the annotation
has pipeline => (is => 'ro', isa => 'ArrayRef[HashRef]');

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

  # TODO: Support sort
  $self->_cleanQuery();

  my ($parentsAref, $childrenAref) = $self->_getHeader();

  my @childrenOrOnly = @$childrenAref;
  my @parentNames = @$parentsAref;

  my @fieldNames = @{$self->fieldNames};
  my $outputHeader = join($delims{fieldSep}, @fieldNames);

  say $outFh $outputHeader;

  if($statsFh) {
    say $statsFh $outputHeader;
  }

  ($err, my $discordantIdx) = $self->_getDiscordantIdx();

  if($err) {
    $self->log('fatal', "Couldn't find discordant index");
  }

  # TODO: figure out why even with field => 'pos' we get very slow perf
  # when having slices > shards with very large (1000+ terms) queries

  my ($nSlices, $batchSize, $timeout) = $self->_getSearchParams();

  my $slice = {
    field => 'pos',
    max => $nSlices,

  };

  my $hasSort = exists $self->inputQueryBody->{sort};
  my $progressFunc = $self->_makeLogProgress($hasSort, $outFh, $statsFh, 3e4);

  my @filterFunctions;
  if($self->pipeline) {
    my ($err, $filters) = $self->_makePipeline();

    if($err) {
      return ($err, undef);
    }

    if(defined $filters) {
      @filterFunctions = @$filters;
    }
  }

  MCE::Loop::init {
    max_workers => $nSlices,
    chunk_size => 1,
    gather => $progressFunc,
  };

  $self->log('info', "Beginning to create annotation from the query");

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
      scroll      => $timeout,
      size        => $batchSize,
      body        => $self->inputQueryBody,
      index       => $self->indexName,
      type        => $self->indexType,
    );

    while(my @docs = $scroll->next($batchSize)) {
      my @sourceData;
      # $#sourceData = $#docs;

      my $skipped = 0;
      DOCS: for my $doc (@docs) {
        if(@filterFunctions) {
          for my $f (@filterFunctions) {
            if($f->($doc->{_source})) {
              $skipped++;

              next DOCS;
            }
          }
        }

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

        push @sourceData, \@rowData;
      }

      my $outputString = _makeOutputString(\@sourceData, \%delims);

      $mce->gather(scalar @docs - $skipped, $outputString, $id);
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

sub _makePipeline {
  my $self = shift;

  state $funcs = {
    binomMaf => \&makeBinomFilter,
    hwe => \&makeHweFilter,
  };

  if(!$self->pipeline) {
    return (undef, undef);
  }

  my @funcs;

  for my $step (@{$self->pipeline}) {
    if(!exists $funcs->{$step->{key}}) {
      $self->log('warn', "Couldn't find function for $step->{key}");
      next;
    }

    my $makeFunc = $funcs->{$step->{key}};

    # Skipped for some reason
    if(!$makeFunc) {
      next;
    }

    push @funcs, $makeFunc->($self, $step);
  }

  return (undef, \@funcs);
}

sub _getSearchParams {
  my $self = shift;

  my $numTerms = $self->_getNumTerms($self->inputQueryBody);

  # -1 simply means we can't approximate the size
  my $nSlices;
  if($numTerms < 0 || $numTerms > 200) {
    $nSlices = $self->shards;
  } else {
    $nSlices = $self->_getSlices();
  }

  my $batchSize = $self->batchSize;

  my $timeout = '2m';

  return ($nSlices, $batchSize, $timeout);
}

sub _cleanQuery {
  my $self = shift;

  # TODO: Support sort
  $self->inputQueryBody->{sort} = ['_doc'];

  if(exists $self->inputQueryBody->{aggs}) {
    delete $self->inputQueryBody->{aggs};
  }

  return;
}

# TODO: Support more versions
sub _getNumTerms {
  my $self = shift;

  my $bool = $self->inputQueryBody->{query}{bool};

  if(!$bool) {
    return -1;
  }

  # TODO: handle complex scripts, which are incredibly slow in ES
  # my $hasScripts;

  # # if($bool->{filter}) {
  # #   my $str = encode_json($bool->{filter});

  # #   if($str =~ /script/isg) {
  # #     $hasScripts = 1;
  # #   }
  # # }

  # # # scripts are very expensive, like large queries
  # # if($hasScripts) {
  # #   return -1;
  # # }

  my $mustObj = $bool->{must} || $bool->{filter};

  my $mustQLen = 0;
  my $mustQuery = '';

  for my $obj (ref $mustObj eq 'ARRAY' ? $mustObj : $mustObj) {
    if($obj->{query_string} && $obj->{query_string}{query}) {
      $mustQLen += split(/\s+/, $obj->{query_string}{query});
      next;
    }

    if($obj->{match}) {
      for my $matchProps (values %{$obj->{match}}) {
        # in the form query: {match: {field: {query: string, operator: string}}
        if(ref $matchProps) {
          $mustQLen += split(/\s+/, $matchProps->{query});
          next;
        }

        # in the form query: {match: {field: string}}
        $mustQLen += split(/\s+/, $matchProps);
      }
    }
  }

  return $mustQLen;
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

sub makeHweFilter {
  my $self = shift;
  my $props = shift;

  my $nSamples = $props->{numSamples};

  if($nSamples < 1) {
    return;
  }

  my $alpha = $props->{critValue};

  if(!$alpha) {
    return;
  }

  # A positive z value
  my $chiCrit = chisqrdistr(1, $alpha);

  # Copy-on-write in multithreaded env; no need to worry about race
  my($eHets, $eHomsMajor, $eHomsMinor, $n, $hets, $homsMajor, $homsMinor, $p);

  # binomial approximation
  # http://www.halotype.com/RKM/figures/TJF/binomial.txt
  return sub {
    #my $doc = shift;
    # $_[0]

    # $q = $_[0]->{'sampleMaf'}[0][0];
    $p = 1 - $_[0]->{'sampleMaf'}[0][0];

    # TODO: should we allow sites like these? Currently skip
    if($p == 0) {
      return 1;
    }

    $n = $nSamples * (1 - $_[0]->{'missingness'}[0][0]);

    $eHets = 2 * $p * (1 - $p) * $n;
    $eHomsMajor = ($p ** 2) * $n;
    $eHomsMinor = $n - ($eHets + $eHomsMajor);
    
    $hets = $n * $_[0]->{'heterozygosity'}[0][0];
    $homsMinor = $n * $_[0]->{'homozygosity'}[0][0];
    $homsMajor = $n - ($hets + $homsMinor);

    if($eHets == 0 || $eHomsMajor == 0 || $eHomsMinor == 0) {
      say STDERR "n: $n, missinginess: $_[0]->{'missingness'}[0][0], nSamples: $nSamples, p: $p, eHets: $eHets ; eHomsMajor : $eHomsMajor; eHomsMinor: $eHomsMinor";
      say STDERR "hets: $hets, homsMajor: $homsMajor; homsMinor: $homsMinor";
      sleep(1000);
    }
    # Returns truthy if test statistic is > $chiCrit, which means 
    # in rejection region == skip
    return $chiCrit < ( ($hets - $eHets) ** 2 ) / $eHets
           + ( ($homsMajor - $eHomsMajor) ** 2) / $eHomsMajor
           + ( ($homsMinor - $eHomsMinor) ** 2) / $eHomsMinor;
  }
}

# TODO: add binomial test if n is small
# Requires numSamples, estimates, and critValue
# Else will not filter anything
sub makeBinomFilter {
  my $self = shift;
  my $props = shift;

  my $nChromosomes = $props->{numSamples} * 2;

  if($nChromosomes < 2) {
    return;
  }

  my $afFieldsAref = $props->{estimates};

  if(!@$afFieldsAref) {
    return;
  }

  my $alpha = $props->{critValue};

  if(!$alpha) {
    return;
  }

  if($alpha == 0 || $alpha > .5) {
    $self->log('error', "Alpha must larger than 0 and smaller than .5");
    return;
  }

  # A positive z value; udistr for .05 will give 1.65
  # unlike Math::Gauss, for which inv_cdf will give -1.65
  my $zCrit = udistr($alpha);

  my $snpOnly = $props->{snpOnly};
  my $privateMaf = $props->{privateMaf} || 0;

  my $minPossibleEstimate =  sprintf("%.2f", 1/$nChromosomes);

  # Different from 1, because of possible rounding
  # my $minK = 1.5;

  # TODO: Don't defeat? If privateMaf is set,
  # then at minimum we should not allow it to be less than $minPossibleEstimate
  if($privateMaf < $minPossibleEstimate) {
    $privateMaf = $minPossibleEstimate;
  }

  # If we calculate K to be a number not quite 1, count it as 1
  # for purpose of checking whether we're at minimum
  my $roundedMinK = 1.1;

  # say STDERR "MIN: $minPossibleEstimate ; snpOnly: $snpOnly ; privateMaf: $minPossibleEstimate ; zCrit: $zCrit; alpha: $alpha";
  # sleep(1000);

  for my $f (@$afFieldsAref) {
    my @path = split(/[.]/, $f);

    $f = \@path;
  }

  # Copy-on-write in multithreaded env; no need to worry about race
  my($n, $k, $p, $isRare);

  # binomial approximation
  # http://www.halotype.com/RKM/figures/TJF/binomial.txt
  return sub {
    #my $doc = shift;
    # $_[0]
                         #$doc->
    if($snpOnly && length($_[0]->{'alt'}[0][0]) > 1) {
      #0 means don't skip
      return 0;
    }

                            #$doc->
    $n = $nChromosomes * (1 - $_[0]->{'missingness'}[0][0]);
             #$doc->
    $k = $n * $_[0]->{'sampleMaf'}[0][0];

    # NOW WE PASS RARE THINGS ONLY IF ESTIMATES ARE ALSO RARE
    # The 2nd condition is to ensure that we don't consider something like 1.0016
    # to be more common that our $minimumPossibleEstimate
    $isRare =  $_[0]->{'sampleMaf'}[0][0] <= $privateMaf || ($n == $nChromosomes && $k < 1.5);

    undef $p;

    my $tested = 0;
    AF_LOOP: for my $field (@{$afFieldsAref}) {
       # Avoid auto-vivification
                #$doc->
      if(!exists $_[0]->{$field->[0]}) {
        next;
      }

      # Example: gnomad.exomes.af
      if(@$field == 3) {
                  #$doc->                                       #$doc->
        if(!exists $_[0]->{$field->[0]}{$field->[1]} || !exists $_[0]->{$field->[0]}{$field->[1]}{$field->[2]}) {
          next AF_LOOP;
        }

            #$doc->
        $p = $_[0]->{$field->[0]}{$field->[1]}{$field->[2]}[0][0];
      } elsif(@$field == 2) {
        # Example: dbSNP.alleleFreq
                   #$doc->
        if(!exists $_[0]->{$field->[0]}{$field->[1]}) {
          next AF_LOOP;
        }

            #$doc->
        $p = $_[0]->{$field->[0]}{$field->[1]}[0][0];
      } else {
        # Try to avoid the expensive loop
        $p = $_[0]->{$field->[0]};

        for(my $i = 1; $i < @$field; $i++) {
          if(!exists $p->{$field->[$i]}) {
            next AF_LOOP;
          }

          $p = $p->{$field->[$i]}
        }

        $p = $p->[0][0];
      }

      if(!defined $p) {
        next AF_LOOP;
      }

      # Handles cases where p == 1 (can't calculate p-value), p == 0, and p <= privateMef
      # TODO: May want to skip to next pop af estimate is $p == 1
      # Currently if $p == 1 we allow only if !isRare
      # Because something that is extremely common in the population
      # but is rare in ours, will be something very very odd
      # or an annotation mistake
      if(($p == 1 && !$isRare) || ($p <= $privateMaf && $isRare)) {
        return 0;
      }

      # say STDERR "z is ". abs($n * $p - $k) / sqrt( $n * $p * (1 - $p) ) . " for p: $p n: $n , k: $k, zCrit: $zCrit";

      # TODO: allow flag to be more conservative; drop if doesn't pass in all
      # TODO: prevent underflow more consistently
      # If we have a smaller deviation than allowed by our critical value
      if( abs($k - $n * $p) / sqrt( $n * $p * (1 - $p) ) <= $zCrit) {
        return 0;
      }

      $tested++;
    }

    # If the site isn't present in then populations of interest it is
    # likely to be rare.
    # In those cases, accept sites thats are rare in our dataset as well
    if($tested == 0 && $isRare) {
      return 0;
    }

    return 1;
}


  # Returns 1 if we want to skip this site, 0 otherwise
  # return sub {
  #    #my ($doc) = @_;
  #    #    $_[0]

  #                         #$doc
  #    if($snpOnly && length($_[0]->{'alt'}[0][0]) > 1) {
  #      #0 means don't skip
  #      return 0;
  #    }
  #      #$doc
  #    if($_[0]->{'sampleMaf'}[0][0] <= $privateMaf) {
  #      return 0;
  #    }
  #                     #$doc
  #    my $n = $N * (1 - $_[0]->{'missingness'}[0][0]);
  #                 #$doc
  #    my $ac = $n * $_[0]->{'sampleMaf'}[0][0];

  #   for my $field (@{$afFieldsAref}) {
  #     my $f = $_[0]->{$field->[0]};

  #     if(@$field == 2) {
  #       $f = $f->{$field->[1]};
  #     } elsif(@$field == 3) {
  #       $f = $f->{$field->[1]}{$field->[2]};
  #     } else {
  #       for(my $i = 1; $i < @$field; $i++) {
  #         $f = $f->{$field->[$i]}
  #       }
  #     }

  #     if(!defined $f) {
  #       next;
  #     }

  #                   #$doc
  #     if(_binomProb($f->[0][0], $n, $ac) >= $alpha) {
  #       return 0;
  #     }
  #   }

  #    # 1 means skip
  #    return 1;
  # }
}

# sub _binomProb {
#   # N is likely the number of chromosomes
#   #my ($popAf, $N, $ac) = @_;
#   #    $_[0]  $_[1], $_[2]

#   #  $popAf       $popAf
#   if($_[0] < 0 || $_[0] > 1) {
#     return 0;
#   }

#   #  $N           $N
#   if($_[1] < 2 || $_[1] > 10_000_000) {
#     return 0;
#   }

#   #           $popAf
#   my $q = 1 - $_[0];
#   #                $N    * $popAf
#   my $cent = round($_[1] * $_[0]);

#   if($cent == 0) {
#     return 0;
#   }

#   p $cent;

#   my @L;
#   #     $N
#   $#L = $_[1];

#   $L[$cent] = 1;

#   my $eps = 1e-8 / $_[1];
#   my $tot = 1;

#   my $k;
#   for(my $i = $cent - 1; $i >= 0; $i--) {
#     $k = $L[$i + 1] * $q * ($i + 1);
#     #     $popAf   $N
#     $k /= $_[0] * ($_[1] - $i);
#     p $k;
#     if($k < $eps) {
#       $L[$i] = 0;
#       $i = 0;
#     } else {
#       $L[$i] = $k;
#     }

#     $tot += $L[$i];
#   }
#   say STDERR "TOT IS $tot";
#   sleep(1000);

#   for(my $i = $cent + 1; $i <= $_[1]; $i++) {
#     #               $popAf   $N
# 	  $k = $L[$i-1] * $_[0] * ($_[1]-($i-1));
# 	  $k /= $q * $i;

# 	  if($k < $eps) {
# 		  $L[$i] = 0;
#       #    $N
# 		  $i = $_[1];
# 	  } else {
# 		  $L[$i] = $k;
# 	  }

# 	  $tot += $L[$i];
#   }
# p @L;
#   #                    $N
#   for(my $i = 0; $i <= $_[1]; $i++) {
#     if(!defined $L[$i]) {
#       say STDERR "NOT DEF WHY";
#       sleep(1000);
#     }
#     $L[$i] /= $tot;
#     #	print "$i $L[$i]\n";
#   }

#   my $rightTail = 0;

#   #           $ac         $N
#   for(my $i = $_[2]; $i<= $_[1]; $i++) {
#     $rightTail += $L[$i];
#   }

#   return $rightTail;
# }

__PACKAGE__->meta->make_immutable;

1;
