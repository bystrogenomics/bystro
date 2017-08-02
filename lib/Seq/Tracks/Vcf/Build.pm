use 5.10.0;
use strict;
use warnings;
package Seq::Tracks::Vcf::Build;

our $VERSION = '0.001';

=head1 DESCRIPTION

  @class Seq::Tracks::Vcf::Build
  Takes a VCF file, runs it through a vcf pre-processor to get it into 
  our internal annotation format, and then uses the info field to build a database

=cut

# TODO: better error handling in the vcf pre-processor
# TODO: Support fields delimited by something other than just , for Number=A
# TODO: Move opening of vcf to Seq::Input::Vcf
use Mouse 2;

use namespace::autoclean;
use List::MoreUtils qw/firstidx/;
use Parallel::ForkManager;
use Scalar::Util qw/looks_like_number/;
use Seq::Output::Delimiters;
use Seq::Tracks::Base::Types;

use DDP;

extends 'Seq::Tracks::Build';

with 'Seq::Output::Fields';
# We assume sparse tracks have at least one feature; can remove this requirement
# But will need to update makeMergeFunc to not assume an array of values (at least one key => value)
has '+features' => (required => 1);

has vcfProcessor => (is => 'ro', isa => 'Str', required => 1);

# We skip entries that span more than this number of bases
has maxVariantSize => (is => 'ro', isa => 'Int', lazy => 1, default => 32);
has keepId => (is => 'ro', isa => 'Bool', lazy => 1, default => !!1);

state $converter = Seq::Tracks::Base::Types->new();
sub BUILD {
  my $self = shift;

  # The length of each feature + 1 character , since 
  # VCF will have these as featureName=
  $self->{_featureLengths} = [ map { length($_) + 1 } @{$self->features} ];
  $self->{_featuresVcfForm} = [ map { "$_\=" } @{$self->features} ];
}

# has altName => (is => '')
sub buildTrack {
  my $self = shift;

  my $pm = Parallel::ForkManager->new($self->max_threads);

  my $outputter = Seq::Output::Delimiters->new();

  my $delim = $outputter->emptyFieldChar;

  for my $file (@{$self->local_files}) {
    $self->log('info', $self->name . ": beginning building from $file");

    $pm->start($file) and next;
      my $echoProg = $self->isCompressedSingle($file) ? $self->gzip . ' -d -c' : 'cat';

      my $wantedChr;
      # Get an instance of the merge function that closes over $self
      # Note that tracking which positinos have been over-written will only work
      # if there is one chromosome per file, or if all chromosomes are in one file
      my $mergeFunc = $self->makeMergeFunc();
      # Record which chromosomes were recorded for completionMeta
      my %visitedChrs;

      my %fieldDbNames;

      my $errPath = $file . ".build." . localtime() . ".log";

      my ($err, @fieldDescriptions) = $self->_extractHeader($file);
      
      if($err) {
        # DB not open yet, no need to commit
        $pm->finish(255, \$err);
      }

      open(my $fh, '-|', "$echoProg $file | " . $self->vcfProcessor . " --emptyField $delim"
        . " --keepId --keepInfo");

      my $count = 0;
      # my ($chr, @fields, @sparseData, $start, $end);
      while ( my $line = $fh->getline() ) {
        my @fields = split '\t', $line;

        #                      (alleleIdx  , info
        my ($err, $valuesAref) = $self->_extractFeatures($fields[-2], $fields[-1], \@fieldDescriptions);

        if($err) {
          #Commit, sync everything, including completion status, and release mmap
          $self->db->cleanUp();
          $pm->finish(255, \$err);
        }

        $count++;
      }

      say "Read $count lines";
    $pm->finish();
  }

  $pm->run_on_finish(sub {
    my ($pid, $exitCode, $fileName, undef, undef, $errRef) = @_;

    if($exitCode != 0) {
      my $err = $errRef ? "due to: $$errRef" : "due to an untimely demise";

      $self->log('fatal', $self->name . ": Failed to build $fileName $err");
      die $self->name . ": Failed to build $fileName $err";
    }

    $self->log('info', $self->name . ": completed building from $fileName");
  });

  $pm->wait_all_children;
}

sub _extractHeader {
  my $self = shift;
  my $file = shift;
  my $dieIfNotFound = shift;

  my @types;

  my $echoProg = $self->isCompressedSingle($file) ? $self->gzip . ' -d -c' : 'cat';

  open(my $fh, '-|', "$echoProg $file");

  my @header;
  while(<$fh>) {
    chomp;

    if(substr($_, 0, 1) eq '#') {
      push @header, $_;
      next;
    }

    last;
  }

  close $fh;

  FEATURE_LOOP: for my $feature (@{$self->features}) {
    for my $h (@header) {
      if(index($h, "INFO\=\<ID\=$feature,") > 0) {
        $h =~ /Number=(\w+)/;

        my $number = $1;

        # In case Number and Type aren't adjacent to each other
        $h =~ /Type=(\w+)/;

        my $type = $1;

        push @types, [$number, $type];

        next FEATURE_LOOP;
      }
    }

    if($dieIfNotFound) {
      return ("Couldn't find $feature, exiting", undef);
    }
  }

  return (undef, @types);
}

sub _extractFeatures {
  my $self = shift;
  # vcfProcessor will split multiallelics
  my $alleleIdx = shift;
  my $info = shift;
  my $fieldDescriptionsAref = shift;

  my @infoFields = split ';', $info;

  my @returnData;

  my $idx = -1;
  my $firstChars;
  FEATURE_LOOP: for my $feature (@{$self->{_featuresVcfForm}}) {
    $idx++;

    for my $field (@infoFields) {
      if(index($field, $feature) > -1) {
        my $val = (split '=', $field)[1];

        if($fieldDescriptionsAref->[$idx][0] eq 'A') {
          my @vals = split ',', $val;

          if(@vals - 1 < $alleleIdx) {
            return ("Err: Number=A field has fewer values than alleles", undef);
          }

          $val = $vals[$alleleIdx];
        }

        # TODO: support non-scalar values
        # TODO: configure from either type specified in YAML, or from VCF Type=
        $val = $self->coerceFeatureType($self->features->[$idx], $val);
       
        push @returnData, $val;

        next FEATURE_LOOP;
      }
    }

    return ("Couldn't find $feature", undef);
    push @returnData, undef;
  }

  return (undef, \@returnData);
}

__PACKAGE__->meta->make_immutable;
1;