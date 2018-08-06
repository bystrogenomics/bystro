package Seq::Output;
use 5.10.0;
use strict;
use warnings;

use Mouse 2;

use Seq::Output::Delimiters;
use Seq::Headers;

with 'Seq::Role::Message';

has header => (is => 'ro', isa => 'Seq::Headers', required => 1);
has delimiters => (is => 'ro', isa => 'Seq::Output::Delimiters', default => sub {
  return Seq::Output::Delimiters->new();
});

sub BUILD {
  my $self = shift;

  $self->{_orderedHeader} = $self->header->getOrderedHeader();
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

  my $missChar = $self->delimiters->emptyFieldChar;
  my $overlapDelim = $self->delimiters->overlapDelimiter;
  my $posDelim = $self->delimiters->positionDelimiter;
  my $valDelim = $self->delimiters->valueDelimiter;
  my $fieldSep = $self->delimiters->fieldSeparator;

  my $trackIdx;
  for my $row (@$outputDataAref) {
    next if !$row;

    $trackIdx = -1;

    TRACK_LOOP: for my $trackName ( @{$self->{_orderedHeader}} ) {
      $trackIdx++;

      # If this track is a parent with children
      if(ref $trackName) {
        if(!defined $row->[$trackIdx] || !@{$row->[$trackIdx]}) {
          $row->[$trackIdx] = join($fieldSep, ($missChar) x @$trackName);

          next TRACK_LOOP;
        }

        for my $featIdx (0 .. $#$trackName) {
            for my $posData (@{$row->[$trackIdx][$featIdx]}) {
              $posData //= $missChar;

              # At this position, the feature is scalar
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

          $row->[$trackIdx][$featIdx] = join($posDelim, @{$row->[$trackIdx][$featIdx]});
        }

        $row->[$trackIdx] = join($fieldSep, @{$row->[$trackIdx]});

        next;
      }

      # Nothing to be done, it's already a scalar
      if(!ref $row->[$trackIdx]) {
        $row->[$trackIdx] //= $missChar;

        next TRACK_LOOP
      }

      if(!defined $row->[$trackIdx] || ref $row->[$trackIdx] && !@{$row->[$trackIdx]}) {
        $row->[$trackIdx] = $missChar;

        next TRACK_LOOP;
      }

      for my $posData (@{$row->[$trackIdx]}) {
        $posData //= $missChar;

        if(ref $posData) {
          $posData = join($valDelim, map { defined $_ ? $_ : $missChar } @$posData);
        }
      }

      $row->[$trackIdx] = join($posDelim, @{$row->[$trackIdx]});
    }

    $row = join("\t", @$row);
  }
  return join("\n", @$outputDataAref) . "\n";
}

__PACKAGE__->meta->make_immutable;
1;