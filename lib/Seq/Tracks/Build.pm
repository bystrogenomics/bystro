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

# Faster than regex trim
use String::Strip qw/StripLTSpace/;

extends 'Seq::Tracks::Base';
# All builders need getReadFh
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
# If this flag is set, then the consumer can choose to skip entire files
# if an unexpected chr is found, or if the expected chr is recorded completed
# Change from b9: this now needs to be manually set, opt-in
has chrPerFile => (is => 'ro', isa => 'Bool', default => 0);

has maxThreads => (is => 'ro', isa => 'Int', lazy => 1, default => 8);
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

  my $d = Seq::Output::Delimiters->new();
  $self->{_cleanDelims} = $d->cleanDelims;
  $self->{_missChar} = $d->emptyFieldChar;
  $self->{_replChar} = $d->globalReplaceChar;
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

    if(!looks_like_number($val)) {
      $_[0]->{_cleanDelims}->($val);
      $_[0]->_stripAndCoerceUndef($val);
    }

    if( defined $type && defined $val ) {
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
    } elsif($leftHand eq '-') {
      $codeRef = sub {
        # my $fieldValue = shift;
        # same as $_[0];

        return $_[0] - $rightHand;
      }
    } elsif($leftHand eq '+') {
      $codeRef = sub {
        # my $fieldValue = shift;
        # same as $_[0];

        return $_[0] + $rightHand;
      }
    } elsif($leftHand eq 'split') {
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

      # commits automatically, so that we are ensured that overlaps
      # called from different threads succeed
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
        # commits automatically, so that we are ensured that overlaps
        # called from different threads succeed
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

# TODO: Allow to be configured on per-track basis
sub _stripAndCoerceUndef {
  #my ($self, $dataStr) = @_;

 # TODO: This will be configurable, per-track
  state $cl = {
    'no assertion provided' => 1,
    'no_assertion_provided' => 1,
    'no assertion criteria provided' => 1,
    'no_assertion_criteria_provided' => 1,
    'no interpretation for the single variant' => 1,
    'no assertion for the individual variant' => 1,
    'no_assertion_for_the_individual_variant' => 1,
    'not provided' => 1,
    'not_provided' => 1,
    'not specified' => 1,
    'not_specified' => 1,
    'see cases' => 1,
    'see_cases' => 1,
    'unknown' => 1
  };

  # STripLTSpace modifies passed string by stripping space from it
  # This modifies the caller's version
  StripLTSpace($_[1]);

  if($_[1] eq '') {
    $_[1] = undef;
    return $_[1];
  }

  my $v = lc($_[1]);

  # These will always get coerced to undef
  # TODO: we may want to force only the missChar comparison
  if($v eq '.' || $v eq 'na' || $v eq $_[0]->{_missChar}){
    $_[1] = undef;
    return $_[1];
  }

  # This will be configurable
  if(exists $cl->{$v}) {
    $_[1] = undef;
    return $_[1];
  }

  return $_[1];
}

sub chrWantedAndIncomplete {
  my ($self, $chr) = @_;

  # Allow users to pass 0 as a valid chromosome, in case coding is odder than we expect
  if(!defined $chr || (!$chr && "$chr" eq '')) {
    return undef;
  }

  if($self->chrIsWanted($chr) && $self->completionMeta->okToBuild($chr)) {
    return $chr;
  }

  return undef;
}

sub safeCloseBuilderFh {
  my ($self, $fh, $fileName, $errCode, $strict) = @_;

  if(!$errCode) {
    $errCode = 'fatal';
  }

  #From Seq::Role::IO
  my $err = $self->safeClose($fh);

  if($err) {
    #Can happen when closing immediately after opening
    if($? != 13) {
      $self->log($errCode, $self->name . ": Failed to close $fileName: $err ($?)");
      return $err;
    }

    # We make a choice to ignored exit code 13... it happens a lot
    # 13 is sigpipe, occurs if closing pipe before cat/pigz finishes
    $self->log('warn', $self->name . ": Failed to close $fileName: $err ($?)");

    # Make it optional to return a sigpipe error, since controlling
    # program likely wants to die on error, and sigpipe may not be worth it
    if($strict) {
      return $err;
    }

    return;
  }

  $self->log('info', $self->name . ": closed $fileName with $?");
  return;
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
