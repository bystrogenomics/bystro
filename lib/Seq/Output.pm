package Seq::Output;
use 5.10.0;
use strict;
use warnings;

use Mouse 2;

use List::Util qw/min max/;

use Seq::Output::Delimiters;
use Seq::Headers;
use DDP;

with 'Seq::Role::Message';

has header => (is => 'ro', isa => 'Seq::Headers', required => 1);
has trackOutIndices => (is => 'ro', isa => 'ArrayRef', required => 1);

has delimiters => (is => 'ro', isa => 'Seq::Output::Delimiters', default => sub {
  return Seq::Output::Delimiters->new();
});

has header => (is => 'ro', isa => 'Seq::Headers', required => 1);

sub BUILD {
  my $self = shift;

  my @trackOutIndices = @{$self->trackOutIndices};
  my $minIdx = min(@trackOutIndices);
  my $maxIdx = max(@trackOutIndices);

  my $fieldSep = $self->delimiters->fieldSeparator;
  my $missChar = $self->delimiters->emptyFieldChar;

  $self->{_trackOutIndices} = [];
  $self->{_trackFeatCount} = [];

  my @header = @{$self->header->getOrderedHeader()};

  # Use array for faster lookup in hot loop
  @{$self->{_trackFeatCount}} = @header;

  my $outIdx = -1;
  for my $trackName ( @header ) {
    $outIdx++;

    if($outIdx < $minIdx || $outIdx > $maxIdx) {
      next;
    }

    push @{$self->{_trackOutIndices}}, $outIdx;

    if(ref $trackName) {
      $self->{_trackFeatCounts}[$outIdx] = $#$trackName;
    } else {
      $self->{_trackFeatCounts}[$outIdx] =  0;
    }
  }
}

# TODO: will be singleton, configured once for all consumers
sub initialize {
}

# ABSTRACT: Knows how to make an output string
# VERSION

#takes an array of <HashRef> data that is what we grabbed from the database
#and whatever else we added to it
#and an array of <ArrayRef> input data, which contains our original input fields
#which we are going to re-use in our output (namely chr, position, type alleles)
sub makeOutputString {
  my ($self, $outputDataAref) = @_;

  # Re-assigning these isn't a big deal beause makeOutputString
  # Called very few times; expected to be called every few thousand rows
  my $missChar = $self->delimiters->emptyFieldChar;
  my $overlapDelim = $self->delimiters->overlapDelimiter;
  my $posDelim = $self->delimiters->positionDelimiter;
  my $valDelim = $self->delimiters->valueDelimiter;
  my $fieldSep = $self->delimiters->fieldSeparator;
  my $featCounts = $self->{_trackFeatCounts};

  for my $row (@$outputDataAref) {
    next if !$row;
    # info = [$outIdx, $numFeatures, $missingValue]
    # if $numFeatures == 0, this track has no features
    TRACK_LOOP: for my $oIdx ( @{$self->{_trackOutIndices}} ) {
      # If this track has no features
      if($featCounts->[$oIdx] == 0) {
        # We always expect output, for any track
        # to be at least a 1 member array
        # because we need to know where in an indel we are
        # or whether we're in a snp
        # So... reference for insance is [A] for a snp
        # and maybe [A, T, C] for a 3 base deletion

        # Most common case, not an indel
        # Currently we have no 0-feature tracks
        if(@{$row->[$oIdx]} == 1) {
          if(!defined $row->[$oIdx][0]) {
            $row->[$oIdx] = $missChar;

            next;
          }

          if(ref $row->[$oIdx][0]) {
            $row->[$oIdx] = join($valDelim, @{$row->[$oIdx][0]});
            next;
          }

          $row->[$oIdx] = $row->[$oIdx][0];

          next;
        }

        # For things without features, we currently support
        # ref (scalar), phastCons, phyloP, cadd, which are all scalars

        # If its not a scalar, its because we have an indel
        # Then, for each position, the thing may be defined, or not
        # It's an array, for instance, CADD scores are
        $row->[$oIdx] = join($posDelim,
          map {
            defined $_
            ? (
                ref $_
                ?
                # at this position this feature has multiple values
                join($valDelim, map { defined $_ ? $_ : $missChar } @$_)
                :
                # at this position this feature has 1 value
                $_
              )
            : $missChar
          } @{$row->[$oIdx]}
        );

        next;
      }

      # If this track is missing altogether it will be an empty array
      # But it will be an array

      for my $featIdx (0 .. $featCounts->[$oIdx]) {
        if(!defined $row->[$oIdx][$featIdx]) {
          $row->[$oIdx][$featIdx] = $missChar;

          next;
        }

        # Typically, we have no indel
        # Which means the feature has only 1 value
        if(@{$row->[$oIdx][$featIdx]} == 1) {
          if(!defined $row->[$oIdx][$featIdx][0]) {
            $row->[$oIdx][$featIdx] = $missChar;
            next;
          }

          # Typically we have a scalar
          if(!ref $row->[$oIdx][$featIdx][0]) {
            $row->[$oIdx][$featIdx] = $row->[$oIdx][$featIdx][0];

            next;
          }

          $row->[$oIdx][$featIdx] = join($valDelim,
            map {
              defined $_
              ? (
                  ref $_
                  ?
                  # at this position this feature has multiple values
                  join($overlapDelim, map { defined $_ ? $_ : $missChar } @$_)
                  :
                  # at this position this feature has 1 value
                  $_
                )
              : $missChar
            } @{$row->[$oIdx][$featIdx][0]}
          );

          next;
        }


        for my $posData (@{$row->[$oIdx][$featIdx]}) {
          if(!defined $posData) {
            $posData = $missChar;

            next;
          }

          # At this position in the indel, value is scalar
          if(!ref $posData) {
            next;
          }


          # At this position, the feature is nested to some degree
          # This is often seen where say one transcript
          # has many descriptive values in one feature
          # Say it has 2 names
          # We want to nest those names, so that we can maintain name/transcript
          # order, ala
          # featureTranscrpipt \t featureTranscriptNames
          # t1;t2 \t t1_name1\\t1_name2;t2_onlyName
          $posData = join($valDelim, map {
            defined $_
            ? (
                ref $_
                ?
                # at this position this feature has multiple values
                join($overlapDelim, map { defined $_ ? $_ : $missChar } @$_)
                :
                # at this position this feature has 1 value
                $_
              )
            : $missChar
          } @$posData);
        }

        $row->[$oIdx][$featIdx] = join($posDelim, @{$row->[$oIdx][$featIdx]});
      }

      # Fields are separated by something like tab
      $row->[$oIdx] = join($fieldSep, @{$row->[$oIdx]});
    }

    # Tracks are also separated by something like tab
    $row = join($fieldSep, @$row);
  }

  return join("\n", @$outputDataAref) . "\n";
}

__PACKAGE__->meta->make_immutable;
1;