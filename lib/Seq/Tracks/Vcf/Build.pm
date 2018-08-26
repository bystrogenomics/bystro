use 5.10.0;
use strict;
use warnings;

package Seq::Tracks::Vcf::Build;

our $VERSION = '0.001';

=head1 DESCRIPTION

  @class Seq::Tracks::Vcf::Build
  Takes a VCF file, runs it through a vcf pre-processor to get it into 
  our internal annotation format, and then uses the info field to build a database
  Will skip discordant sites, logging them
=cut

# TODO: better error handling in the vcf pre-processor
# TODO: Support fields delimited by something other than just , for Number=A
# TODO: Move opening of vcf to Seq::Input::Vcf
# TODO: Be more explicit with naming of indices from the intermediate annotation output
# (such as vcfAltIdx = 4, rather than using 4 itself)

# Note ALT field is required, if not found will be appended
use Mouse 2;

use namespace::autoclean;
use List::MoreUtils qw/firstidx/;
use Parallel::ForkManager;
use Scalar::Util qw/looks_like_number/;
use Seq::Output::Delimiters;
use Seq::Tracks::Base::Types;
use Scalar::Util qw/looks_like_number/;
use DDP;

use Seq::Tracks;
extends 'Seq::Tracks::Build';

with 'Seq::Output::Fields';

# We assume sparse tracks have at least one feature; can remove this requirement
# But will need to update makeMergeFunc to not assume an array of values (at least one key => value)
has '+features' => (required => 1);

# Like Sparse tracks are typically quite small, with potentiall quite large values
# so to make optimal use of pages
# lets set a smaller default commitEvery
has '+commitEvery' => (default => 1e3);

has vcfProcessor => (is => 'ro', isa => 'Str', required => 1, default => 'bystro-vcf');

state $converter = Seq::Tracks::Base::Types->new();

# Defines the indices expected in the intermediate vcf output of
# $self->vcfProcessor
# TODO: read these values from the header of a single file
state $vcfFeatures = {
  chrom => 0, pos => 1, type => 2, ref => 3, alt => 4, trTv => 5,
  heterozygotes => 6, homozygotes => 8, missingGenos => 10, id => 15
};

sub BUILD {
  my $self = shift;

  my $features = $self->features;

  if(!@{$features}) {
    $self->log('fatal', 'VCF build requires INFO features');
  }

  my %featuresMap;

  for(my $i = 0; $i < @{$features}; $i++) {
    $featuresMap{lc($features->[$i])} = $i;
  }

  if(!defined $featuresMap{alt}) {
    $self->log('fatal', $self->name . ": 'alt' feature not specified, required for vcf tracks");
  }

  my %fieldMap = map{ lc($_) => $self->fieldMap->{$_} } keys %{$self->fieldMap};

  my %visitedVcfFeatures;
  my @headerFeatures;
  for my $vcfFeature (keys %$vcfFeatures) {
    my $idx;

    if($visitedVcfFeatures{$vcfFeature}) {
      $self->log('fatal', "Duplicate feature requested: $vcfFeature");
    }

    $visitedVcfFeatures{$vcfFeature} = 1;

    # Because VCF files are so flexible with feature definitions, it will be
    # difficult to tell if a certain feature just isn't present in a vcf file
    # Easier to make feature definition flexible, especially since one 
    # may correctly surmise that we read the VCF after transformation to intermediate
    # annotated format
    my $lcVcfFeature = lc($vcfFeature);

    if(defined $featuresMap{$lcVcfFeature}) {
      $idx = $featuresMap{$lcVcfFeature};
    } elsif(defined $fieldMap{$lcVcfFeature} && defined $featuresMap{$fieldMap{$lcVcfFeature}}) {
      $idx = $featuresMap{$fieldMap{$lcVcfFeature}};
    }

    # This $vcfFeature isn't requested by the user
    if(!defined $idx) {
      next;
    }

    #Stores:
    #1) The feature name (post-transformation)
    #2) The index in the intermedaite annotation file
    #3) The index in the database
    push @headerFeatures, [
      $self->features->[$idx], $vcfFeatures->{$vcfFeature},
      $self->getFieldDbName($self->features->[$idx])
    ];
  }

  # We could also force-add alt; would get properly inserted into db.
  # However, we would reduce confidence users had in the representation stated
  # in the YAML config
  if(!defined $visitedVcfFeatures{alt}) {
    $self->log('fatal', "alt (or ALT) field is required for vcf tracks, used to match input alleles");
  }

  $self->{_headerFeatures} = \@headerFeatures;

  my %reverseFieldMap = map { $self->fieldMap->{$_} => $_ } keys %{$self->fieldMap};

  my %infoFeatureNames;
  for my $feature (@{$self->features}) {
    my $originalName = $reverseFieldMap{$feature} || $feature;

    # skip the first few columns, don't allow ALT in INFO
    if(defined $visitedVcfFeatures{lc($originalName)}) {
      next;
    }

    $infoFeatureNames{$feature} = $originalName;
  }

  # TODO: prevent header features from overriding
  $self->{_infoFeatureNames} = \%infoFeatureNames;
  $self->{_numFilters} = scalar keys %{$self->build_row_filters} || 0;

  # Precalculate the field db names, for faster accesss
  # TODO: think about moving away from storing the "db name" in the database
  # We may just want to enforce no changs to the order of fields once
  # The db is created
  # It fails in too many ways; for instance if you remove a feature,
  # Then try to build again, it will crash, because expected array length
  # shorter than some of the remaining field indices stored in db, potentially
  my %fieldDbNames;

  for my $feature (@{$self->features}) {
    $fieldDbNames{$feature} = $self->getFieldDbName($feature);
  }

  $self->{_fieldDbNames} = \%fieldDbNames;

  my $tracks = Seq::Tracks->new();
  $self->{_refTrack} = $tracks->getRefTrackGetter();

  # TODO: Read bystro-vcf header, and configure $vcfFeatures based on that
  # will require either reading the first file in the list, or giving
  # bystro-vcf a "output only the header" feature (but scope creep)
}

# has altName => (is => '')
sub buildTrack {
  my $self = shift;

  my $pm = Parallel::ForkManager->new($self->maxThreads);

  my $outputter = Seq::Output::Delimiters->new();

  my $delim = $outputter->emptyFieldChar;

  # my $altIdx = $self->headerFeatures->{ALT};
  # my $idIdx = $self->headerFeatures->{ID};

  my $lastIdx = $#{$self->features};

  # location of these features in input file (intermediate annotation)
  my $refIdx = $vcfFeatures->{ref};
  my $posIdx = $vcfFeatures->{pos};
  my $chrIdx = $vcfFeatures->{chrom};
  my $altIdx = $vcfFeatures->{alt};

  # Track over-written positions
  # Hashes all values passed in, to make sure that duplicate values aren't written
  my ($mergeFunc, $cleanUpMerge) = $self->makeMergeFunc();

  my %completedDetails;
  $pm->run_on_finish(sub {
    my ($pid, $exitCode, $fileName, undef, undef, $errOrChrs) = @_;

    if($exitCode != 0) {
      my $err = $errOrChrs ? "due to: $$errOrChrs" : "due to an untimely demise";

      $self->log('fatal', $self->name . ": Failed to build $fileName $err");
    }

    # TODO: check for hash ref
    for my $chr (keys %$errOrChrs) {
      if(!$completedDetails{$chr}) {
        $completedDetails{$chr} = [$fileName];
      } else {
        push @{$completedDetails{$chr}}, $fileName;
      }
    }

    $self->log('info', $self->name . ": completed building from $fileName");
  });

  for my $file (@{$self->local_files}) {
    $self->log('info', $self->name . ": beginning building from $file");

    # Although this should be unnecessary, environments must be created
    # within the process that uses them
    # This provides a measure of safety
    $self->db->cleanUp();

    $pm->start($file) and next;
      my $prog = $self->isCompressedSingle($file) ? $self->gzip . ' ' . $self->decompressArgs : 'cat';

      my $errPath = $file . ".build." . localtime() . ".log";

      my ($err, $vcfNameMap, $vcfFilterMap) = $self->_extractHeader($file);

      if($err) {
        # DB not open yet, no need to commit
        $pm->finish(255, \$err);
      }

       # Record which chromosomes were recorded for completionMeta
      my %visitedChrs;
      my $chr;
      my @fields;
      my $dbData;
      my $wantedChr;
      my $refExpected;
      my $dbPos;

      # We use "unsafe" writers, whose active count we need to track
      my $cursor;
      my $count = 0;

      my $op = "$prog $file | " . $self->vcfProcessor. " --emptyField $delim" . " --keepId --keepInfo";
      $err = $self->safeOpen(my $fh, '-|', $op);

      if($err) {
        $self->log('fatal', $self->name . ": $err");
      }

      # TODO: Read header, and configure vcf header feature indices based on that
      my $header = <$fh>;

      FH_LOOP: while ( my $line = $fh->getline() ) {
        chomp $line;
        # This is the annotation input first 7 lines, plus id, info
        @fields = split '\t', $line;

        $chr = $fields[$chrIdx];

        # falsy value is ''
        if(!defined $wantedChr || $wantedChr ne $chr) {
          # We have a new chromosome
          if(defined $wantedChr) {
            #Commit any remaining transactions, remove the db map from memory
            #this also has the effect of closing all cursors
            $self->db->cleanUp();
            undef $cursor;

            $count = 0;
          }

          $wantedChr = $self->chrWantedAndIncomplete($chr);
        }

        # TODO: rethink chPerFile handling
        if(!defined $wantedChr) {
          next FH_LOOP;
        }

        $visitedChrs{$wantedChr} //= 1;

        # 0-based position: VCF is 1-based
        $dbPos = $fields[$posIdx] - 1;

        if(!looks_like_number($dbPos)) {
          $self->db->cleanUp();

          $pm->finish(255, \"Invalid position @ $chr\: $dbPos");
        }

        $cursor //= $self->db->dbStartCursorTxn($wantedChr);

        # We want to keep a consistent view of our universe, so use one transaction
        # during read/modify/write
        $dbData = $self->db->dbReadOneCursorUnsafe($cursor, $dbPos);

        $refExpected = $self->{_refTrack}->get($dbData);
        if($fields[$refIdx] ne $refExpected) {
          $self->log('warn', $self->name . " $chr\:$fields[$posIdx]: "
            . " Discordant. Expected ref: $refExpected, found: ref: $fields[$refIdx], alt:$fields[$altIdx]. Skipping");
          next;
        }

        ($err, my $data) = $self->_extractFeatures(\@fields, $vcfNameMap, $vcfFilterMap);

        if($err) {
          #Commit, sync everything, including completion status, and release mmap
          $self->db->cleanUp();

          $pm->finish(255, \$err);
        }

        # If the row didn't pass filters, $data will be undefined
        # In all other cases it will be an array
        if(!defined $data) {
          next;
        }

        #Args:                         $cursor, $chr,       $trackIndex,   $pos,   $trackValue, $mergeFunction
        $self->db->dbPatchCursorUnsafe($cursor, $wantedChr, $self->dbName, $dbPos, $data, $mergeFunc);

        if($count > $self->commitEvery) {
          $self->db->dbEndCursorTxn($wantedChr);
          undef $cursor;

          $count = 0;
        }

        $count++;
      }

      #Commit, sync everything, including completion status, and release mmap
      $self->db->cleanUp();

      $self->safeCloseBuilderFh($fh, $file, 'fatal');

    $pm->finish(0, \%visitedChrs);
  }

  $pm->wait_all_children();

  for my $chr (keys %completedDetails) {
    $self->completionMeta->recordCompletion($chr);

    # cleanUpMerge placed here so that only after all files are processed do we
    # drop the temporary merge databases
    # so that if we have out-of-order chromosomes, we do not mishandle
    # overlapping sites
    $cleanUpMerge->($chr);

    $self->log('info', $self->name . ": recorded $chr completed, from "
    . (join(",", @{$completedDetails{$chr}})));
  }

  #TODO: figure out why this is necessary, even with DEMOLISH
  $self->db->cleanUp();
  return;
}

sub _extractHeader {
  my $self = shift;
  my $file = shift;
  my $dieIfNotFound = shift;

  my ($err, undef, $fh) = $self->getReadFh($file);

  if($err) {
    return ($err, undef, undef);
  }

  my @header;
  while(<$fh>) {
    chomp;

    if(substr($_, 0, 1) eq '#') {
      push @header, $_;
      next;
    }

    last;
  }

  $err = $self->safeCloseBuilderFh($fh, $file, 'error');

  if($err) {
    return ($err, undef, undef);
  }

  my $idxOfInfo = -9;
  my $idx = -1;

  my %nameMap;
  my %filterMap;

  # Flags may or may not be in the info field
  # To speed search, store these, and walk back to find our value
  my $flagCount = 0;
  for my $h (@header) {
    $idx++;

    if($h !~ /\#\#INFO=/) {
      next;
    }

    if($idxOfInfo == -9) {
      $idxOfInfo = $idx;
    }

    $h =~ /Number=([\w.]+)/;

    my $number = $1;

    $h =~ /Type=(\w+)/;

    my $type = $1;

    # Keep track of things that look like they could mess up INFO string order
    # Flag in particular seems often missing, so we'll do a linear search
    # From $idx - $idxOfInfo to +$flagCount
    if(looks_like_number($number)) {
      if($number == 0) {
         $flagCount++;
      }
    } elsif($number eq '.') {
      $flagCount++;
    }

    my $featIdx = -1;

    # TODO: if the flag item is the feature we're searching for do something
    # Not critial, but will have less efficient search
    # Requires precise spelling of the vcf feature
    # TODO: Die if don't find header for any requested feature
    FEATURE_LOOP: for my $feature (@{$self->features}) {
      if(!defined $self->{_infoFeatureNames}{$feature}) {
        next;
      }

      my $infoName = $self->{_infoFeatureNames}{$feature};

      if(index($h, "INFO\=\<ID\=$infoName,") > 0) {
        # my $vcfName = "$feature=";
        # In case Number and Type aren't adjacent to each other
        # $return[$featIdx] = [$number, $type];
        $nameMap{$infoName} = [$feature, $number, $type, $idx, ];
        last FEATURE_LOOP;
      }
    }

    # Filters on INFO fields
    FEATURE_LOOP: for my $feature (keys %{$self->build_row_filters}) {
      my $infoName = $self->{_infoFeatureNames}{$feature} || $feature;

      if(index($h, "INFO\=\<ID\=$infoName,") > 0) {
        # my $vcfName = "$feature=";
        # In case Number and Type aren't adjacent to each other
        # $return[$featIdx] = [$number, $type];
        $filterMap{$infoName} = [$feature, $number, $type, $idx, ];
        last FEATURE_LOOP;
      }
    }
  }

  return (undef, \%nameMap, \%filterMap);
}

sub _extractFeatures {
  my ($self, $fieldsAref, $vcfNameMap, $vcfFilterMap, $fieldDbNames) = @_;
  
  # vcfProcessor will split multiallelics, store the alleleIdx
  # my @infoFields = ;

  my @returnData;
  $#returnData = $#{$self->features};

  my $firstChars;

  my $warned;

  my $entry;
  my $found = 0;
  my $name;
  my $val;

  my $totalNeeded = @returnData + $self->{_numFilters};
  # $arr holds
  # 1) field name
  # 2) index in intermediate annotation
  # 3) index in database
  for my $arr (@{$self->{_headerFeatures}}) {
    # $arr->[0] is the fieldName
    # $arr->[1] is the field idx
    if($self->hasTransform($arr->[0]) ) {
      $fieldsAref->[$arr->[1]] = $self->transformField($arr->[0], $fieldsAref->[$arr->[1]]);
    }

    $returnData[$arr->[2]] = $self->coerceFeatureType($arr->[0], $fieldsAref->[$arr->[1]]);
  }

  my $alleleIdx = $fieldsAref->[-2];

  for my $info (split ';', $fieldsAref->[-1]) {
    # If # found == scalar @{$self->features}
    if($found == $totalNeeded) {
      last;
    }

    $name = substr($info, 0, index($info, '='));

    $entry = $vcfNameMap->{$name} || $vcfFilterMap->{$name};

    # p $entry;
    if(!$entry) {
      next;
    }

    $found++;

    $val = substr($info, index($info, '=') + 1);

    # A types have a value per allele
    if($entry->[1] eq 'A') {
      my @vals = split ',', $val;

      if(@vals - 1 < $alleleIdx) {
        return ("Err: Type=A field has fewer values than alleles", undef);
      }

      $val = $vals[$alleleIdx];
    }

    # Using $entry->[0] allows us to map the name of the property to be filtered
    if($self->hasFilter($entry->[0])) {
      if(!$self->passesFilter($entry->[0], $val)) {
        return (undef, undef);
      }

      next;
    }

    # All field to be split if the user requests that in the YAML config
    # $entry->[0] is the fieldName
    if($self->hasTransform($entry->[0]) ) {
      $val = $self->transformField($entry->[0], $val);
    }

    # TODO: support non-scalar values
    # TODO: configure from either type specified in YAML, or from VCF Type=
    $returnData[$self->{_fieldDbNames}{$entry->[0]}] = $self->coerceFeatureType($entry->[0], $val);
  }

  return (undef, \@returnData);
}

__PACKAGE__->meta->make_immutable;
1;