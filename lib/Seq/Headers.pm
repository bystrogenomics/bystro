package Seq::Headers;
use Mouse 2;

# # Abstract: Responsible for building the header object and string
use 5.10.0;
use strict;
use warnings;
use namespace::autoclean;
use DDP;
use List::Util qw/first/;

with 'Seq::Role::Message';
#stored as array ref to preserve order
# [ { $parent => [ $child1, $child2 ] }, $feature2, $feature3, etc ]
state $orderedHeaderFeaturesAref = [];
# [ [ $child1, $child2 ], $feature2, $feature3, etc ]
state $orderedHeaderFeaturesArefNoMap = [];
# { $parent => [ $child1, $child2 ] }
state $parentChild = {};
# { $parent => {childName => childIdx}}
state $parentChildHashRef = {};
# { childFeature1 => idx, childFeature2 => idx;
state $orderMap = {};

# All singleton tracks have an initialize method, which clears 
sub initialize() {
  $orderedHeaderFeaturesAref = [];
  $orderedHeaderFeaturesArefNoMap = [];
  $parentChild = {};
  $parentChildHashRef = {};
  $orderMap = {};
}

sub get() {
  return $orderedHeaderFeaturesAref;
}

sub getParentFeatures {
  my ($self, $parentName) = @_;
  return $parentChild->{$parentName};
}

sub getChildFeaturesMap {
  my ($self, $parentName) = @_;

  if($parentChildHashRef->{$parentName}) {
    return $parentChildHashRef->{$parentName};
  }

  my %map;
  for my $i (0 .. $#$orderedHeaderFeaturesAref) {
    # say "trackName is $orderedHeaderFeaturesAref->[$i]";
    if(ref $orderedHeaderFeaturesAref->[$i]) {
      my $trackName = (keys %{$orderedHeaderFeaturesAref->[$i]})[0];

      # say "trackName is $trackName, parentName is $parentName";
      if($trackName eq $parentName) {
        my $y = 0;
        for my $child (@{ $orderedHeaderFeaturesAref->[$i]{$trackName} }) {
          # say "child is $child, id is $y";
          $map{$child} = $y;
          $y++;
        }
      }

      next;
    }
  }

  $parentChildHashRef->{$parentName} = %map ? \%map : undef;

  return $parentChildHashRef->{$parentName}; 
}

# Memoized, should be called only after all features of interest are added

sub getOrderedHeaderNoMap() {
  if(@$orderedHeaderFeaturesArefNoMap) {
    return $orderedHeaderFeaturesArefNoMap;
  }

  for my $i (0 .. $#$orderedHeaderFeaturesAref) {
    if(ref $orderedHeaderFeaturesAref->[$i]) {
      my $trackName = (keys %{$orderedHeaderFeaturesAref->[$i]})[0];

      $orderedHeaderFeaturesArefNoMap->[$i] = $orderedHeaderFeaturesAref->[$i]{$trackName};
    } else {
      $orderedHeaderFeaturesArefNoMap->[$i] = $orderedHeaderFeaturesAref->[$i];
    }
  }

  return $orderedHeaderFeaturesArefNoMap; 
}

# Memoized, should be called only after all features of interest are added
sub getParentFeaturesMap() {
  if(%$orderMap) {
    return $orderMap;
  }

  for my $i (0 .. $#$orderedHeaderFeaturesAref) {
    if(ref $orderedHeaderFeaturesAref->[$i]) {
      $orderMap->{ (keys %{$orderedHeaderFeaturesAref->[$i]})[0] } = $i;
    } else {
      $orderMap->{$orderedHeaderFeaturesAref->[$i]} = $i;
    }
  }

  return $orderMap; 
}

sub getString {
  my $self = shift;

  my @out;  
  for my $feature (@$orderedHeaderFeaturesAref) {
    #this is a parentName => [$feature1, $feature2, $feature3] entry
    if(ref $feature) {
      my ($parentName) = %$feature;
      foreach (@{ $feature->{$parentName} } ) {
        push @out, "$parentName.$_";
      }
      next;
    }
    push @out, $feature;
  }

  return join("\t", @out);
}

#######################addFeaturesToHeader#######################
# Description: Add a single feature to the header
# @param <Str|Number> $child: A feature name (required)
# @param <Str|Number> $parent: A parent name that the $child belongs to (optional)
# @param <Any> $prepend: Whether or not to add the $child to the beginning of
# the features array, or to the beginning of the $parent feature array if !!$parent
sub addFeaturesToHeader {
  my ($self, $child, $parent, $prepend) = @_;

  if(ref $child eq 'ARRAY') {
    goto &_addFeaturesToHeaderBulk;
  }

  if($parent) {
    my $parentFound = 0;

    for my $headerEntry (@$orderedHeaderFeaturesAref) {
      if(!ref $headerEntry) {
        if($parent eq $headerEntry) {
          $self->log('warning', "$parent equals $headerEntry, which has no 
            child features, which was not what we expected");
        }
        next;
      }

      my ($key, $valuesAref) = %$headerEntry;

      if($key eq $parent) {
        # If we have already added this feature, exit the function
        if(defined(first {$_ eq $child} @$valuesAref)) {
          return;
        }

        if($prepend) {
          unshift @$valuesAref, $child;
        } else {
          push @$valuesAref, $child;
        }

        $parentChild->{$parent} = $valuesAref;

        return;
      }
    }

    # No parent found, no need to check if feature has previously been added
    my $val = { $parent => [$child] };

    if($prepend) {
      unshift @$orderedHeaderFeaturesAref, $val;
    } else {
      push @$orderedHeaderFeaturesAref, $val;
    }

    $parentChild->{$parent} = [$child];

    return;
  }

  ######## No parent provided;  we expect that the child is the only ##########
  ####### value stored, rather than a parentName => [value1, value2] ##########

  # If the value was previously added, exit function;
  if( defined(first {$_ eq $child} @$orderedHeaderFeaturesAref) ) {
    return;
  }

  if($prepend) {
    unshift @$orderedHeaderFeaturesAref, $child;
  } else {
    push @$orderedHeaderFeaturesAref, $child;
  }
}

sub _addFeaturesToHeaderBulk {
  my ($self, $childrenAref, $parent, $prepend) = @_;

  if(!ref $childrenAref) {
    goto &addFeaturesToHeader;
  }

  my @array = $prepend ? reverse @$childrenAref : @$childrenAref;

  for my $child (@array) {
    $self->addFeaturesToHeader($child, $parent, $prepend);
  }

  return;
}

__PACKAGE__->meta->make_immutable;

1;