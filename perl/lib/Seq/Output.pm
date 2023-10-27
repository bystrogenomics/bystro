package Seq::Output;
use 5.10.0;
use strict;
use warnings;

use Mouse 2;

use List::Util qw/min max/;

use Seq::Output::Delimiters;
use Seq::Headers;

with 'Seq::Role::Message';

has header          => ( is => 'ro', isa => 'Seq::Headers', required => 1 );
has trackOutIndices => ( is => 'ro', isa => 'ArrayRef',     required => 1 );

has delimiters => (
  is      => 'ro',
  isa     => 'Seq::Output::Delimiters',
  default => sub {
    return Seq::Output::Delimiters->new();
  }
);

has refTrackName => (
  is  => 'ro',
  isa => 'Str',
);

sub BUILD {
  my $self = shift;

  my @trackOutIndices = @{ $self->trackOutIndices };
  my $minIdx          = min(@trackOutIndices);
  my $maxIdx          = max(@trackOutIndices);

  # Cache delimiters to avoid method calls in hot loop
  $self->{_emptyFieldChar} = $self->delimiters->emptyFieldChar;
  $self->{_overlapDelim}   = $self->delimiters->overlapDelimiter;
  $self->{_valDelim}       = $self->delimiters->valueDelimiter;

  $self->{_trackOutIndices} = [];
  $self->{_trackFeatCount}  = [];

  my @header = @{ $self->header->getOrderedHeader() };

  # Use array for faster lookup in hot loop
  @{ $self->{_trackFeatCount} } = @header;

  my $outIdx = -1;
  for my $trackName (@header) {
    $outIdx++;

    if ( $outIdx < $minIdx || $outIdx > $maxIdx ) {
      next;
    }

    push @{ $self->{_trackOutIndices} }, $outIdx;

    if ( $trackName eq $self->refTrackName ) {
      $self->{_refTrackIdx} = $outIdx;
      next;
    }

    if ( ref $trackName ) {
      $self->{_trackFeatCounts}[$outIdx] = $#$trackName;
    }
    else {
      $self->{_trackFeatCounts}[$outIdx] = 0;
    }
  }
}

sub uniqueify {
  my %count;
  my $undefCount = 0;

  if ( !@{ $_[0] } ) {
    return $_[0];
  }

  foreach my $value ( @{ $_[0] } ) {
    if ( !defined $value ) {
      $undefCount++;
    }
    else {
      $count{$value} = 1;
    }
  }

  if ( $undefCount == @{ $_[0] } ) {
    return [ $_[0]->[0] ];
  }

  if ( $undefCount == 0 && scalar keys %count == 1 ) {
    return [ $_[0]->[0] ];
  }

  return $_[0];
}

sub mungeRow {
  # $_[0] = $self
  # $_[1] = $row

  for my $row ( @{ $_[1] } ) {
    if ( !defined $row ) {
      $row = $_[0]->{_emptyFieldChar};
      next;
    }

    if ( ref $row ) {
      $row = join(
        $_[0]->{_overlapDelim},
        map { defined $_ ? $_ : $_[0]->{_emptyFieldChar} } @{ uniqueify($row) }
      );
    }
  }

  return join $_[0]->{_valDelim}, @{ uniqueify( $_[1] ) };
}

# ABSTRACT: Knows how to make an output string
# VERSION

#takes an array of <HashRef> data that is what we grabbed from the database
#and whatever else we added to it
#and an array of <ArrayRef> input data, which contains our original input fields
#which we are going to re-use in our output (namely chr, position, type alleles)
sub makeOutputString {
  my ( $self, $outputDataAref ) = @_;

  # Re-assigning these isn't a big deal beause makeOutputString
  # Called very few times; expected to be called every few thousand rows
  my $missChar   = $self->delimiters->emptyFieldChar;
  my $posDelim   = $self->delimiters->positionDelimiter;
  my $fieldSep   = $self->delimiters->fieldSeparator;
  my $featCounts = $self->{_trackFeatCounts};

  for my $row (@$outputDataAref) {
    next if !$row;
    # info = [$outIdx, $numFeatures, $missingValue]
    # if $numFeatures == 0, this track has no features
    TRACK_LOOP: for my $oIdx ( @{ $self->{_trackOutIndices} } ) {
      if ( $oIdx == $self->{_refTrackIdx} ) {
        $row->[$oIdx] = join '', @{ $row->[$oIdx] };
        next;
      }

      # If this track has no features
      if ( $featCounts->[$oIdx] == 0 ) {
        # We always expect output, for any track
        # to be at least a 1 member array
        # because we need to know where in an indel we are
        # or whether we're in a snp
        # So... reference for insance is [A] for a snp
        # and maybe [A, T, C] for a 3 base deletion

        # Most common case, not an indel
        # Currently we have no 0-feature tracks
        if ( @{ $row->[$oIdx] } == 1 ) {
          if ( !defined $row->[$oIdx][0] ) {
            $row->[$oIdx] = $missChar;
            next;
          }

          if ( ref $row->[$oIdx][0] ) {
            $row->[$oIdx] = $self->mungeRow( $row->[$oIdx][0] );
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
        $row->[$oIdx] = join(
          $posDelim,
          @{
            uniqueify(
              [
                map { !defined $_ ? $missChar : ref $_ ? $self->mungeRow($_) : $_ }
                  @{ $row->[$oIdx] }
              ]
            )
          }
        );

        next;
      }

      # If this track is missing altogether it will be an empty array
      # But it will be an array

      for my $featIdx ( 0 .. $featCounts->[$oIdx] ) {
        if ( !defined $row->[$oIdx][$featIdx] ) {
          $row->[$oIdx][$featIdx] = $missChar;
          next;
        }

        # Typically, we have no indel
        # Which means the feature has only 1 value
        if ( @{ $row->[$oIdx][$featIdx] } == 1 ) {
          if ( !defined $row->[$oIdx][$featIdx][0] ) {
            $row->[$oIdx][$featIdx] = $missChar;
            next;
          }

          # Typically we have a scalar
          if ( !ref $row->[$oIdx][$featIdx][0] ) {
            $row->[$oIdx][$featIdx] = $row->[$oIdx][$featIdx][0];
            next;
          }

          $row->[$oIdx][$featIdx] = $self->mungeRow( $row->[$oIdx][$featIdx][0] );
          next;
        }

        for my $posData ( @{ $row->[$oIdx][$featIdx] } ) {
          if ( !defined $posData ) {
            $posData = $missChar;
            next;
          }

          # At this position in the indel, value is scalar
          if ( !ref $posData ) {
            next;
          }

          $posData = $self->mungeRow($posData);
        }

        $row->[$oIdx][$featIdx] =
          join( $posDelim, @{ uniqueify( $row->[$oIdx][$featIdx] ) } );
      }

      # Fields are separated by something like tab
      $row->[$oIdx] = join( $fieldSep, @{ $row->[$oIdx] } );
    }

    # Tracks are also separated by something like tab
    $row = join( $fieldSep, @$row );
  }

  return join( "\n", @$outputDataAref ) . "\n";
}

__PACKAGE__->meta->make_immutable;
1;
