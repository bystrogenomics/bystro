package Seq::Headers;
use Mouse 2;

# # Abstract: Responsible for building the header object and string
use 5.10.0;
use strict;
use warnings;
use namespace::autoclean;
use List::Util qw/first/;

with 'Seq::Role::Message';

#stored as array ref to preserve order
# [ { $parent => [ $child1, $child2 ] }, $feature2, $feature3, etc ]
state $orderedHeaderFeaturesAref = [];

# { $parent => [ $child1, $child2 ] }
state $parentChild = {};

# [ [ $child1, $child2 ], $feature2, $feature3, etc ]
state $orderedHeaderCache = [];
state $strHeaderCache     = '';

# { childFeature1 => idx, childFeature2 => idx;
state $orderMapCache = {};

# { $parent => { $child1 => idxChild1, $child2 => idxChild2 }}
my %parentChildHash;

# All singleton tracks have an initialize method, which clears
sub initialize {
  _clearCache();

  $orderedHeaderFeaturesAref = [];
  $parentChild               = {};

  return;
}

sub _clearCache {

  # These get initialize/cleared every time feature added
  # They simply track different views of
  $orderedHeaderCache = [];
  $orderMapCache      = {};
  undef %parentChildHash;
  $strHeaderCache = '';

  return;
}

sub get {
  return $orderedHeaderFeaturesAref;
}

sub getParentFeatures {
  my ( $self, $parentName ) = @_;
  return $parentChild->{$parentName};
}

sub getFeatureIdx {
  my ( $self, $parentName, $childName ) = @_;

  if ( !%parentChildHash ) {
    my $i = -1;
    for my $entry ( values @$orderedHeaderFeaturesAref ) {
      $i++;

      if ( ref $entry ) {

        # One key only, the parent name (trackName)
        my ($trackName) = keys %{$entry};

        my %children;
        my $y = -1;
        for my $childName ( @{ $entry->{$trackName} } ) {
          $y++;
          $children{$childName} = $y;
        }

        $parentChildHash{$trackName} = \%children;
        next;
      }

      $parentChildHash{'_masterBystro_'} //= {};
      $parentChildHash{'_masterBystro_'}{$entry} = $i;
    }
  }

  $parentName ||= '_masterBystro_';
  return $parentChildHash{$parentName}{$childName};
}

sub getOrderedHeader() {
  if (@$orderedHeaderCache) {
    return $orderedHeaderCache;
  }

  for my $i ( 0 .. $#$orderedHeaderFeaturesAref ) {
    if ( ref $orderedHeaderFeaturesAref->[$i] ) {
      my $trackName = ( keys %{ $orderedHeaderFeaturesAref->[$i] } )[0];

      $orderedHeaderCache->[$i] = $orderedHeaderFeaturesAref->[$i]{$trackName};
    }
    else {
      $orderedHeaderCache->[$i] = $orderedHeaderFeaturesAref->[$i];
    }
  }

  return $orderedHeaderCache;
}

# Retrieves child feature
sub getParentIndices() {
  if (%$orderMapCache) {
    return $orderMapCache;
  }

  for my $i ( 0 .. $#$orderedHeaderFeaturesAref ) {
    if ( ref $orderedHeaderFeaturesAref->[$i] ) {
      $orderMapCache->{ ( keys %{ $orderedHeaderFeaturesAref->[$i] } )[0] } = $i;
    }
    else {
      $orderMapCache->{ $orderedHeaderFeaturesAref->[$i] } = $i;
    }
  }

  return $orderMapCache;
}

sub getString {
  my $self = shift;

  if ($strHeaderCache) {
    return $strHeaderCache;
  }

  my @out;
  for my $feature (@$orderedHeaderFeaturesAref) {

    #this is a parentName => [$feature1, $feature2, $feature3] entry
    if ( ref $feature ) {
      my ($parentName) = %$feature;
      foreach ( @{ $feature->{$parentName} } ) {
        push @out, "$parentName.$_";
      }
      next;
    }
    push @out, $feature;
  }

  $strHeaderCache = join( "\t", @out );
  return $strHeaderCache;
}

#######################addFeaturesToHeader#######################
# Description: Add a single feature to the header
# @param <Str|Number> $child: A feature name (required)
# @param <Str|Number> $parent: A parent name that the $child belongs to (optional)
# @param <Any> $prepend: Whether or not to add the $child to the beginning of
# the features array, or to the beginning of the $parent feature array if !!$parent
sub addFeaturesToHeader {
  my ( $self, $child, $parent, $prepend ) = @_;

  _clearCache();

  if ( ref $child eq 'ARRAY' ) {
    goto &_addFeaturesToHeaderBulk;
  }

  if ($parent) {
    my $parentFound = 0;

    for my $headerEntry (@$orderedHeaderFeaturesAref) {
      if ( !ref $headerEntry ) {
        if ( $parent eq $headerEntry ) {
          $self->log(
            'warning', "$parent equals $headerEntry, which has no
            child features, which was not what we expected"
          );
        }
        next;
      }

      my ( $key, $valuesAref ) = %$headerEntry;

      if ( $key eq $parent ) {

        # If we have already added this feature, exit the function
        if ( defined( first { $_ eq $child } @$valuesAref ) ) {
          return;
        }

        if ($prepend) {
          unshift @$valuesAref, $child;
        }
        else {
          push @$valuesAref, $child;
        }

        $parentChild->{$parent} = $valuesAref;

        return;
      }
    }

    # No parent found, no need to check if feature has previously been added
    my $val = { $parent => [$child] };

    if ($prepend) {
      unshift @$orderedHeaderFeaturesAref, $val;
    }
    else {
      push @$orderedHeaderFeaturesAref, $val;
    }

    $parentChild->{$parent} = [$child];

    return;
  }

  ######## No parent provided;  we expect that the child is the only ##########
  ####### value stored, rather than a parentName => [value1, value2] ##########

  # If the value was previously added, exit function;
  if ( defined( first { $_ eq $child } @$orderedHeaderFeaturesAref ) ) {
    return;
  }

  if ($prepend) {
    unshift @$orderedHeaderFeaturesAref, $child;
  }
  else {
    push @$orderedHeaderFeaturesAref, $child;
  }

  return;
}

sub _addFeaturesToHeaderBulk {
  my ( $self, $childrenAref, $parent, $prepend ) = @_;

  if ( !ref $childrenAref ) {
    goto &addFeaturesToHeader;
  }

  my @array = $prepend ? reverse @$childrenAref : @$childrenAref;

  for my $child (@array) {
    $self->addFeaturesToHeader( $child, $parent, $prepend );
  }

  return;
}

__PACKAGE__->meta->make_immutable;

1;
