package Seq::Output;
use 5.10.0;
use strict;
use warnings;

use Mouse 2;

use Seq::Output::Delimiters;
use Seq::Headers;

with 'Seq::Role::Message';

use DDP;

# TODO: Configure as singleton

has delimiters => (is => 'ro', isa => 'Seq::Output::Delimiters', default => sub {
  return Seq::Output::Delimiters->new();
});

sub BUILD {
  my $self = shift;

  $self->{_headers} = Seq::Headers->new();
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
  # my $fieldSeparator = $self->{_fieldSeparator};
  my $emptyFieldChar = $self->delimiters->emptyFieldChar;

  my $rowIdx;

  my $alleleDelimiter = $self->delimiters->alleleDelimiter;
  my $positionDelimiter = $self->delimiters->positionDelimiter;
  my $valueDelimiter = $self->delimiters->valueDelimiter;
  my $fieldSeparator = $self->delimiters->fieldSeparator;

  if(!$self->{_multiDepth}) {
    my @headers = @{ $self->{_headers}->getOrderedHeaderNoMap() };

    $self->{_multiDepth} = { map {
      $_ => ref $headers[$_] ? 3 : 2;
    } 0 .. $#headers };

    $self->{_orderedHeader} = \@headers;
  }

  my $trackIdx = -1;
  my $multiallelic;
  my $featureData;
  for my $row (@$outputDataAref) {
    next if !$row;

    $rowIdx = 0;
    $trackIdx = 0;

    TRACK_LOOP: for my $trackName ( @{$self->{_orderedHeader}} ) {
      if(ref $trackName) {
        if(!defined $row->[$trackIdx] || ! @{$row->[$trackIdx]}) {
          $row->[$trackIdx] = join($fieldSeparator, ($emptyFieldChar) x @$trackName);

          $trackIdx++;
          next TRACK_LOOP;
        }

        for my $featureIdx (0 .. $#$trackName) {
          for my $alleleData (@{$row->[$trackIdx][$featureIdx]}) {
            for my $positionData (@$alleleData) {
              $positionData //= $emptyFieldChar;

              if(ref $positionData) {
                $positionData = join($valueDelimiter, map { 
                  defined $_
                  # Unfortunately, prior to 11/30/16 Bystro dbs would merge sparse tracks
                  # incorrectly, resulting in an extra array depth
                  ? (ref $_ ? join($valueDelimiter, map { defined $_ ? $_ : $emptyFieldChar } @$_) : $_)
                  : $emptyFieldChar
                } @$positionData);
              }
            }
            $alleleData = @$alleleData > 1 ? join($positionDelimiter, @$alleleData) : $alleleData->[0];
          }

          # p  $row->[$trackIdx][$featureIdx];
          $row->[$trackIdx][$featureIdx] =
            @{$row->[$trackIdx][$featureIdx]} > 1 
            ? join($alleleDelimiter, @{$row->[$trackIdx][$featureIdx]})
            : $row->[$trackIdx][$featureIdx][0];
        }

        $row->[$trackIdx] = join($fieldSeparator, @{$row->[$trackIdx]});

      } else {
        # Nothing to be done, it's already a scalar
        if(!ref $row->[$trackIdx]) {
          $row->[$trackIdx] //= $emptyFieldChar;

          $trackIdx++;
          next TRACK_LOOP
        }

        if(!defined $row->[$trackIdx] || ref $row->[$trackIdx] && !@{$row->[$trackIdx]}) {
          $row->[$trackIdx] = $emptyFieldChar;

          $trackIdx++;
          next TRACK_LOOP;
        }

        for my $alleleData (@{$row->[$trackIdx]}) {
          if(!defined $alleleData) {
            $alleleData = $emptyFieldChar;
            next;
          }

          for my $positionData (@$alleleData) {
            $positionData //= $emptyFieldChar;

            if(ref $positionData) {
              $positionData = join($valueDelimiter, map { defined $_ ? $_ : $emptyFieldChar } @$positionData);
            }
          }

          $alleleData = @$alleleData > 1 ? join($positionDelimiter, @$alleleData) : $alleleData->[0];
        }

        $row->[$trackIdx] = join($alleleDelimiter, @{$row->[$trackIdx]});
      }

      $trackIdx++;
    }
    
    $row = join("\t", @$row);
  }

  return join("\n", @$outputDataAref) . "\n";
}

__PACKAGE__->meta->make_immutable;
1;