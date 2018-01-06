use 5.10.0;
use strict;
use warnings;

package Seq::Tracks::Build;

our $VERSION = '0.001';

# ABSTRACT: A base class for Tracks::*:BUILD classes
# VERSION

use Mouse 2;
use MouseX::NativeTraits;
use namespace::autoclean;
use Scalar::Util qw/looks_like_number/;
use DDP;

use Seq::DBManager;
use Seq::Tracks::Build::CompletionMeta;
use Seq::Tracks::Base::Types;
use Seq::Tracks::Build::LocalFilesPaths;
use Seq::Output::Delimiters;

extends 'Seq::Tracks::Base';
# All builders need get_read_fh
with 'Seq::Role::IO';

#################### Instance Variables #######################################
############################# Public Exports ##################################
has skipCompletionCheck => (is => 'ro');

# Every builder needs access to the database
# Don't specify types because we do not allow consumers to set this attribute
has db => (is => 'ro', init_arg => undef, default => sub {
  return Seq::DBManager->new();
});

# Allows consumers to record track completion, skipping chromosomes that have 
# already been built
has completionMeta => (is => 'ro', init_arg => undef, default => sub { my $self = shift;
  return Seq::Tracks::Build::CompletionMeta->new({name => $self->name,
    db => $self->db, skipCompletionCheck => $self->skipCompletionCheck});
});

# Transaction size. If large, re-use of pages may be inefficient
# https://github.com/LMDB/lmdb/blob/mdb.master/libraries/liblmdb/lmdb.h
has commitEvery => (is => 'rw', isa => 'Int', lazy => 1, default => 2e4);

# All tracks want to know whether we have 1 chromosome per file or not
has chrPerFile => (is => 'ro', init_arg => undef, writer => '_setChrPerFile');

has max_threads => (is => 'ro', isa => 'Int', lazy => 1, default => 8);
########## Arguments taken from YAML config file or passed some other way ##############

#################################### Required ###################################
has local_files => (
  is      => 'ro',
  isa     => 'ArrayRef',
  traits  => ['Array'],
  handles => {
    noLocalFiles => 'is_empty',
    allLocalFiles => 'elements',
  },
  required => 1,
);

########################### Optional arguments ################################
#called based because that's what UCSC calls it
#most things are 0 based, including anything in bed format from UCSC, fasta files
has based => ( is => 'ro', isa => 'Int', default => 0, lazy => 1);

# If a row has a field that doesn't pass this filter, skip it
has build_row_filters => (
  is => 'ro',
  isa => 'HashRef',
  traits => ['Hash'],
  handles => {
    hasFilter => 'exists',
    allFieldsToFilterOn => 'keys',
  },
  lazy => 1,
  default => sub { {} },
);

# Transform a field in some way
has build_field_transformations => (
  is => 'ro',
  isa => 'HashRef',
  traits => ['Hash'],
  handles => {
    hasTransform => 'exists',
    allFieldsToTransform => 'keys',
  },
  lazy => 1,
  default => sub { {} },
);

# The user can rename any input field, this will be used for the feature name
# This makes it possible to store any name in the db, output file, in place
# of the field name in the source file used to make the db
# if fieldMap isn't specified, this property  will be filled with featureName => featureName
has fieldMap => (is => 'ro', isa => 'HashRef', lazy => 1, default => sub {
  my $self = shift;
  my %data = map { $_ => $_ } @{$self->features};

  return \%data;
});

# TODO: config output;
has _emptyFieldRegex => (is => 'ro', isa => 'RegexpRef', init_arg => undef, default => sub { 
  my $delim = Seq::Output::Delimiters->new();

  my $emptyField = $delim->emptyFieldChar;

  my $regex = qr/^\s*$emptyField\s*$/;

  return $regex;
});

################################ Constructor ################################
sub BUILD {
  my $self = shift;

  my @allLocalFiles = $self->allLocalFiles;

  #exported by Seq::Tracks::Base
  my @allWantedChrs = $self->allWantedChrs;

  if(@allWantedChrs > @allLocalFiles && @allLocalFiles > 1) {
    $self->log("warn", "You're specified " . scalar @allLocalFiles . " file for "
      . $self->name . ", but " . scalar @allWantedChrs . " chromosomes. We will "
      . "assume there is only one chromosome per file, and that 1 chromosome isn't accounted for.");
  }

  $self->_setChrPerFile(@allLocalFiles > 1 ? 1 : 0);

  # Commit, sync, and remove any databases opened
  # This is useful because locking may occur if there is an open transaction
  # before fork(), and to make sure that any database meta data is properly
  # committed before tracks begin to use that data.
  Seq::DBManager::cleanUp();
}

# Configure local_files as abs path, and configure required field (*_field_name)
# *_field_name is a computed attribute that the consumer may choose to implement
# Example. In config: 
#  required_field_map:
##   chrom : Chromosome
# We pass on to classes that extend this: 
#   chrom_field_name with value "Chromosome"
my $localFilesHandler = Seq::Tracks::Build::LocalFilesPaths->new();
around BUILDARGS => sub {
  my ($orig, $class, $href) = @_;

  my %data = %$href;

  if(!$href->{files_dir}) {
    $class->log('fatal', "files_dir required for track builders");
  }

  $data{local_files} = $localFilesHandler->makeAbsolutePaths($href->{files_dir},
    $href->{name}, $href->{local_files});

  return $class->$orig(\%data);
};

#########################Type Conversion, Input Field Filtering #########################
#type conversion; try to limit performance impact by avoiding unnec assignments
#@params {String} $_[1] : feature the user wants to check
#@params {String} $_[2] : data for that feature
#@returns {String} : coerced type

# This is stored in Build.pm because this only needs to happen during insertion into db
state $converter = Seq::Tracks::Base::Types->new();
sub coerceFeatureType {
  #my ($self, $feature, $data) = @_;
  # $self == $_[0] , $feature == $_[1], $data == $_[2]

  my $type = $_[0]->getFeatureType( $_[1] );

  # say "$type";
  # Don't mutate the input if no type is stated for the feature
  # if( !defined $type ) {
  #   return $_[2];
  # }

  #### All values sent to coerceFeatureType at least get an undefined check ####

  # modifying the value here actually modifies the value in the array
  # http://stackoverflow.com/questions/2059817/why-is-perl-foreach-variable-assignment-modifying-the-values-in-the-array
  # https://ideone.com/gjWQeS
  for my $val (ref $_[2] ? @{ $_[2] } : $_[2]) {
    if(!defined $val) {
      next;
    }

    $val = $_[0]->coerceUndefinedValues($val);

    if( defined $type ) {
      $val = $converter->convert($val, $type);
    }
  }

  # In order to allow fields to be well-indexed by ElasticSearch or other engines
  # and to normalize delimiters in the output, anything that has a comma
  # (or whatever multi_delim set to), return as an array reference
  return $_[2];
}

sub passesFilter {
  state $cachedFilters;

  if( $cachedFilters->{$_[1]} ) {
    return &{ $cachedFilters->{$_[1]} }($_[2]);
  }

  #   $_[0],      $_[1],    $_[2]
  my ($self, $featureName, $featureValue) = @_;

  my $command = $self->build_row_filters->{$featureName};

  my ($infix, $value) = split(' ', $command);

  if ($infix eq '==') {
    if(looks_like_number($value) ) {
      $cachedFilters->{$featureName} = sub {
        my $fieldValue = shift;

        return $fieldValue == $value; 
      } 
    } else {
      $cachedFilters->{$featureName} = sub {
        my $fieldValue = shift;

        return $fieldValue eq $value; 
      }
    }
  } elsif ($infix eq '!=') {
    if(looks_like_number($value) ) {
      $cachedFilters->{$featureName} = sub {
        my $fieldValue = shift;

        return $fieldValue != $value;
      }
    } else {
      $cachedFilters->{$featureName} = sub {
        my $fieldValue = shift;

        return $fieldValue ne $value;
      }
    }
  } elsif($infix eq '>') {
    $cachedFilters->{$featureName} = sub {
      my $fieldValue = shift;
      return $fieldValue > $value;
    }
  } elsif($infix eq '>=') {
    $cachedFilters->{$featureName} = sub {
      my $fieldValue = shift;
      return $fieldValue >= $value;
    }
  } elsif ($infix eq '<') {
    $cachedFilters->{$featureName} = sub {
      my $fieldValue = shift;
      return $fieldValue < $value;
    }
  } elsif ($infix eq '<=') {
    $cachedFilters->{$featureName} = sub {
      my $fieldValue = shift;
      return $fieldValue <= $value;
    }
  } else {
    $self->log('warn', "This filter, ".  $self->build_row_filters->{$featureName} . 
      ", uses an  operator $infix that isn\'t supported.
      Therefore this filter won\'t be run, and all values for $featureName will be allowed");
    #allow all
    $cachedFilters->{$featureName} = sub { return 1; };
  }

  return &{ $cachedFilters->{$featureName} }($featureValue);
}

######################### Field Transformations ###########################
#TODO: taint check the modifying value
state $transformOperators = ['.', 'split', '-', '+'];
sub transformField {
  state $cachedTransform;

  if( defined $cachedTransform->{$_[0]->name}{$_[1]} ) {
    return &{ $cachedTransform->{$_[0]->name}{$_[1]} }($_[2]);
  }

  #   $_[0],      $_[1],    $_[2]
  my ($self, $featureName, $featureValue) = @_;

  my $command = $self->build_field_transformations->{$featureName};

  my ($leftHand, $rightHand) = split(' ', $command);

  my $codeRef;

  if($self->_isTransformOperator($leftHand) ) {
    if($leftHand eq '.') {
      $codeRef = sub {
        # my $fieldValue = shift;
        # same as $_[0];

        return $_[0] . $rightHand;
      }
    }

    if($leftHand eq '-') {
      $codeRef = sub {
        # my $fieldValue = shift;
        # same as $_[0];

        return $_[0] - $rightHand;
      }
    }

    if($leftHand eq '+') {
      $codeRef = sub {
        # my $fieldValue = shift;
        # same as $_[0];

        return $_[0] + $rightHand;
      }
    }

    if($leftHand eq 'split') {
      $codeRef = sub {
        # my $fieldValue = shift;
        # same as $_[0];
        my @out;

        # if trailing ,; or whichever specified delimiter
        # remove so that no trailing undef value remains
        $_[0] =~ s/\s*$rightHand\s*$//;

        # Some fields may contain no data after the delimiter,
        # which will lead to blank data, don't keep that
        # TODO: skipping empty fields is dangerous; may lead to data that is
        # ordered to fall out of order
        # evalute the choice on line 349
        foreach(split(/$rightHand/, $_[0]) ) {
          # Remove trailing/leading whitespace
          $_ =~ s/^\s+//;
          $_ =~ s/\s+$//;

          if(defined $_ && $_ ne '') {
            push @out, $_;
          }
        }

        return @out == 1 ? $out[0] : \@out;
      }
    }
  } elsif($self->_isTransformOperator($rightHand) ) {
    # Append text in the other direction
    if($rightHand eq '.') {
      $codeRef = sub {
       # my $fieldValue = shift;
       # same as $_[0];
        return $leftHand . $_[0];
      }
    }

    # Don't allow +/- as right hand operator, pointless and a little silly
  }

  if(!defined $codeRef) {
    $self->log('warn', "Requested transformation, $command, for $featureName, not understood");
    return $featureValue;
  }

  $cachedTransform->{$self->name}{$featureName} = $codeRef;

  return &{$codeRef}($featureValue);
}

# Merge [featuresOld...] with [featuresNew...]
# Expects 2 arrays of equal length
# //Won't merge when [featuresNew...] previously merged (duplicate)
# Note: it is completely unsafe to dbReadOne and not commit here
# If the user relies on a non DB->Txn transactions, LMDB_File will complain
# that the transaction should be a sub-transaction
# may screw up the parent transaction, since we currently use a single
# transaction per database per thread. We should move away from this.
# TODO: allow 2 scalars, or growing one array to match the lenght of the other
# TODO:  dupsort, dupfixed to optimize storage
# TODO: get reliable de-duping algorithm of deep structures
sub makeMergeFunc {
  my $self = shift;

  my $name = $self->name;
  my $madeIntoArray = {};

  my $tempDbName = "$name\_merge_temp";
  return ( sub {
      my ($chr, $pos, $oldTrackVal, $newTrackVal) = @_;

      if(!ref $newTrackVal || @$newTrackVal != @$oldTrackVal) {
        return("makeMergeFunc accepts only array values of equal length", undef);
      }

      my $seen = $self->db->dbReadOne("$tempDbName/$chr", $pos);

      my @updated;
      $#updated = $#$oldTrackVal;

      # oldTrackVal and $newTrackVal should both be arrays, with at least one index
      for (my $i = 0; $i < @$newTrackVal; $i++) {
        if(!$seen) {
          $updated[$i] = [$oldTrackVal->[$i], $newTrackVal->[$i]];
          next;
        }

        $updated[$i] = [@{$oldTrackVal->[$i]}, $newTrackVal->[$i]];
      }

      if(!$seen) {
        $self->db->dbPut("$tempDbName/$chr", $pos, 1);
      }

      return (undef, \@updated);
    },

    sub {
      my $chr = shift;
      $self->db->dbDropDatabase("$tempDbName/$chr", 1);

      say STDERR "Cleaned up $tempDbName/$chr";
    }
  );
}

sub coerceUndefinedValues {
  #my ($self, $dataStr) = @_;

  # Don't waste storage space on NA. In Bystro undef values equal NA (or whatever
  # Output.pm chooses to represent missing data as.

  if($_[1] =~ /^\s*NA\s*$/i || $_[1] =~/^\s*$/ || $_[1] =~/^\s*\.\s*$/ || $_[1] =~ $_[0]->_emptyFieldRegex) {
    return undef;
  }

  return $_[1];
}

sub _isTransformOperator {
  my ($self, $value) = @_;

  for my $operator (@$transformOperators) {
    if(index($value, $operator) > -1 ) {
      return 1;
    }
  }
  return 0;
}

__PACKAGE__->meta->make_immutable;

1;
