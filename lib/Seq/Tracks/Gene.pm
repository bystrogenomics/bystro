use 5.10.0;
use strict;
use warnings;
package Seq::Tracks::Gene;

our $VERSION = '0.001';

=head1 DESCRIPTION

  @class B<Seq::Gene>
  
  Note: unlike previous Bystro, there is no longer a genomic_type
  Just a siteType, which is Intronic, Coding, 5UTR, etc
  This is just like Annovar's
  We add "Intergenic" if not covered by any gene

  This class also handles intergenic sites

=cut

## TODO: remove the if(nearestGeneNumber) check. Right now needed because
## we have no chrM refSeq stuff
use Mouse 2;

use namespace::autoclean;
use DDP;

extends 'Seq::Tracks::Get';
with 'Seq::Tracks::Region::RegionTrackPath';

use Seq::Tracks::Gene::Site;
use Seq::Tracks::Gene::Site::SiteTypeMap;
use Seq::Tracks::Gene::Site::CodonMap;
use Seq::Tracks::Gene::Definition;
use Seq::DBManager;

########### @Public attributes##########
########### Additional "features" that we will add to our output ##############
### Users may configure these ####

# These are features defined by Gene::Site, but we name them in Seq::Tracks::Gene
# Because it gets really confusing to track down the features defined in Seq::Tracks::Gene::Site
# TODO: rename these siteTypeField to match the interface used by Seq.pm (TODO: and Seq::Tracks::Sparse::Build)
has siteTypeKey => (is => 'ro', default => 'siteType');
has strandKey => (is => 'ro', default => 'strand');
has codonNumberKey => (is => 'ro', default => 'codonNumber');
has codonPositionKey => (is => 'ro', default => 'codonPosition');

has codonSequenceKey => (is => 'ro', default => 'refCodon');
has refAminoAcidKey => (is => 'ro', default => 'refAminoAcid');
has newCodonKey => (is => 'ro', default => 'altCodon');
has newAminoAcidKey => (is => 'ro', default => 'altAminoAcid');
has exonicAlleleFunctionKey => (is => 'ro', default => 'exonicAlleleFunction');

########################## Private Attributes ##################################
########## The names of various features. These cannot be configured ##########
### Positions that aren't covered by a refSeq record are intergenic ###
state $intergenic = 'intergenic';

### txEffect possible values ###
# TODO: export these, and make them configurable
state $silent = 'synonymous';
state $replacement = 'nonSynonymous';
state $frameshift = 'indel-frameshift';
state $inFrame = 'indel-nonFrameshift';
state $indelBoundary = 'indel-exonBoundary';
state $startLoss = 'startLoss';
state $stopLoss = 'stopLoss';
state $stopGain = 'stopGain';

# TODO: implement the truncated annotation
state $truncated = 'truncatedCodon';


### objects that get used by multiple subs, but shouldn't be public attributes ###
# All of these instantiated classes cannot be configured at instantiation time
# so safe to use in static context
state $siteUnpacker = Seq::Tracks::Gene::Site->new();
state $siteTypeMap = Seq::Tracks::Gene::Site::SiteTypeMap->new();
state $codonMap = Seq::Tracks::Gene::Site::CodonMap->new();

state $strandIdx = $siteUnpacker->strandIdx;
state $siteTypeIdx = $siteUnpacker->siteTypeIdx;
state $codonSequenceIdx = $siteUnpacker->codonSequenceIdx;
state $codonPositionIdx = $siteUnpacker->codonPositionIdx;
state $codonNumberIdx = $siteUnpacker->codonNumberIdx;

state $negativeStrandTranslation = { A => 'T', C => 'G', G => 'C', T => 'A' };

### Set the features that we get from the Gene track region database ###
has '+features' => (
  default => sub { 
    my $geneDef = Seq::Tracks::Gene::Definition->new();
    return [$geneDef->allUCSCgeneFeatures, $geneDef->txErrorName]; 
  },
);

#### Add our other "features", everything we find for this site ####
sub BUILD {
  my $self = shift;

  # Private variables, meant to cache often used data
  $self->{_allCachedDbNames} = {};
  $self->{_regionDb} = {};

  $self->{_features} = $self->features;
  $self->{_dbName} = $self->dbName;
  $self->{_db} = Seq::DBManager->new();

  # Not including the txNumberKey;  this is separate from the annotations, which is 
  # what these keys represent

  #  Prepend some custom features
  #  Providing 1 as the last argument means "prepend" instead of append
  #  So these features will come before any other refSeq.* features
  $self->headers->addFeaturesToHeader([
    $self->siteTypeKey, $self->exonicAlleleFunctionKey,
    $self->codonSequenceKey, $self->newCodonKey, $self->refAminoAcidKey,
    $self->newAminoAcidKey, $self->codonPositionKey,
    $self->codonNumberKey, $self->strandKey,
  ], $self->name, 1);

  if($self->hasJoin) {
    my $joinTrackName = $self->joinTrackName;

    $self->{_hasJoin} = 1;
    
    $self->{_flatJoinFeatures} = [map{ "$joinTrackName.$_" } @{$self->joinTrackFeatures}];
    $self->headers->addFeaturesToHeader($self->{_flatJoinFeatures}, $self->name);

    # TODO: Could theoretically be overwritten by line 114
    #the features specified in the region database which we want for nearest gene records
    my $i = 0;
    for my $fName ( @{$self->joinTrackFeatures} ) {
      $self->{_allCachedDbNames}{$self->{_flatJoinFeatures}[$i]} = $self->getFieldDbName($fName);
      $i++;
    }
  }

  for my $fName (@{$self->{_features}}) {
    $self->{_allCachedDbNames}{$fName} = $self->getFieldDbName($fName);
  }

  my @allGeneTrackFeatures = @{ $self->headers->getParentFeatures($self->name) };
  
  # This includes features added to header, using addFeatureToHeader 
  # such as the modified nearest feature names ($nTrackPrefix.$_) and join track names
  # and siteType, strand, codonNumber, etc.
  for my $i (0 .. $#allGeneTrackFeatures) {
    $self->{_featureIdxMap}{ $allGeneTrackFeatures[$i] } = $i;
  }

  $self->{_lastFeatureIdx} = $#allGeneTrackFeatures;
  $self->{_featIdx} = [ 0 .. $#allGeneTrackFeatures ];

  $self->{_strandFidx} = $self->{_featureIdxMap}{$self->strandKey};
  $self->{_siteFidx} = $self->{_featureIdxMap}{$self->siteTypeKey};
  # Avoid accessor penalties by aliasing to the $self hash
  # These correspond to all of the sites held in Gene::Site
  $self->{_codonFIdx} = $self->{_featureIdxMap}{$self->codonSequenceKey};
  $self->{_codonPosFidx} = $self->{_featureIdxMap}{$self->codonPositionKey};
  $self->{_codonNumFidx} = $self->{_featureIdxMap}{$self->codonNumberKey};

  # The values for these keys we calculate at get() time.
  $self->{_refAaFidx} = $self->{_featureIdxMap}{$self->refAminoAcidKey};
  $self->{_altCodonFidx} = $self->{_featureIdxMap}{$self->newCodonKey};
  $self->{_altAaFidx} = $self->{_featureIdxMap}{$self->newAminoAcidKey};
  $self->{_alleleFuncFidx} = $self->{_featureIdxMap}{$self->exonicAlleleFunctionKey};
};

sub get {
  #my ($self, $href, $chr, $refBase, $allele, $alleleIdx, $positionIdx, $outAccum) = @_;
  #    $_[0], $_[1], $_[1], $_[3],   $_[4],   $_[5],      $_[6]         $_[7]
  # WARNING: If $_[1]->[$_[0]->{_dbName} isn't defined, will be treated as the 0 index!!!
  # therefore return here if that is the case
  # ~1/2 of sites will have no gene track entry (including all non-coding, 2% coding)
  if(!defined $_[1]->[$_[0]->{_dbName}]) {
    for my $i (@{$_[0]->{_featIdx}}) {
      $_[7]->[$i][$_[6]] = undef;
    }

    $_[7]->[$_[0]->{_siteFidx}][$_[5]][$_[6]] = $intergenic;
    return $_[7];
  }

  my ($self, $href, $chr, $refBase, $allele, $alleleIdx, $positionIdx, $outAccum) = @_;

  my @out;
  # Set the out array to the size we need; undef for any indices we don't add here
  $#out = $self->{_lastFeatureIdx};

  # Cached field names to make things easier to read
  my $cachedDbNames = $self->{_allCachedDbNames};
  my $idxMap = $self->{_featureIdxMap};

  ################# Cache track's region data ##############
  # returns an array
  $self->{_regionDb}{$chr} //= $self->{_db}->dbReadAll( $self->regionTrackPath($chr) );

  my $geneDb = $self->{_regionDb}{$chr};

  ####### Get all transcript numbers, and site data for this position #########

  #<ArrayRef> $unpackedSites ; <ArrayRef|Int> $txNumbers
  my ($siteData, $txNumbers, $multiple);

  #Reads:
  # ( $href->[$self->{_dbName}] ) {
  if(defined $href->[$self->{_dbName}] ) {
    ($txNumbers, $siteData) = $siteUnpacker->unpack($href->[$self->{_dbName}]);
    $multiple = ref $txNumbers ? $#$txNumbers : 0;
  }
  
  if( !defined $txNumbers ) {
    $out[$_[0]->{_siteFidx}] = $intergenic;
    
    return accumOut($alleleIdx, $positionIdx, $outAccum, \@out);
  }

  if($self->{_hasJoin}) {
    # For join tracks, use only the entry for the first of multiple transcripts
    # Because the data stored is always identical at one position
    my $num = $multiple ? $txNumbers->[0] : $txNumbers;
    # http://ideone.com/jlImGA
    for my $fName ( @{$self->{_flatJoinFeatures}} ) {
      $out[ $idxMap->{$fName} ] = $geneDb->[$num]{$cachedDbNames->{$fName}};
    }
  }

  ################## Populate site information ########################
  # save unpacked sites, for use in txEffectsKey population #####
  # moose attrs are very slow, cache
  # Push, because we'll use the indexes in calculating alleles
  # TODO: Better handling of truncated codons
  # Avoid a bunch of \;\ for non-coding sites
  # By not setting _codonNumberKey, _codonPositionKey, _codonSequenceKey if !hasCodon
  my $hasCodon;
  if($multiple) {
    for my $site (@$siteData) {
      push @{ $out[$self->{_strandFidx}] }, $site->[$strandIdx];
      push @{ $out[$self->{_siteFidx}] }, $site->[$siteTypeIdx];

      if(defined $site->[$codonSequenceIdx]) {
        $hasCodon //= 1;
      }
    }
  } else {
    $out[$self->{_strandFidx}] = $siteData->[$strandIdx];
    $out[$self->{_siteFidx}] = $siteData->[$siteTypeIdx];

    if(defined $siteData->[$codonSequenceIdx]) {
      $hasCodon = 1;
    }
  }
  # ################# Populate geneTrack's user-defined features #####################
  #Reads:            $self->{_features}
  if($multiple) {
    for my $feature (@{$self->{_features}}) {
      $out[$idxMap->{$feature}] = [map { $geneDb->[$_]{$cachedDbNames->{$feature}} } @$txNumbers];
    }
  } else {
    for my $feature (@{$self->{_features}}) {
      $out[$idxMap->{$feature}] = $geneDb->[$txNumbers]{$cachedDbNames->{$feature}};
    }
  }

  # If we want to be ~ 20-50% faster, move this before the Populate Gene Tracks section
  if(!$hasCodon) {
    return accumOut($alleleIdx, $positionIdx, $outAccum, \@out);
  }

  ######Populate _codon*Key, exonicAlleleFunction, amion acids keys ############

  my ($i, @funcAccum, @codonNum, @codonSeq, @codonPos, @refAA, @newAA, @newCodon);
  # Set undefs for every position, other than the ones we need
  # So that we don't need to push undef's to keep transcript order
  $#funcAccum = $#codonNum = $#codonSeq = $#codonPos = $#refAA = $#newAA = $#newCodon = $multiple;

  $i = 0;

  if(length($allele) > 1) {
    # Indels get everything besides the _*AminoAcidKey and _newCodonKey
    my $indelAllele = 
      substr($allele, 0, 1) eq '+'
      ? length(substr($allele, 1)) % 3 ? $frameshift : $inFrame
      : int($allele) % 3 ? $frameshift : $inFrame; 

    for my $site ($multiple ? @$siteData : $siteData) {
      $codonNum[$i] = $site->[$codonNumberIdx];
      $codonSeq[$i] = $site->[$codonSequenceIdx];

      if(defined $site->[$codonSequenceIdx]) {
        $funcAccum[$i] = $indelAllele;
        
        # Codon position only exists (and always does) when codonSequence does
        # We store codonPosition as 0-based, users probably expect 1 based
        $codonPos[$i] = $site->[$codonPositionIdx] + 1;

        if(length($site->[$codonSequenceIdx]) == 3) {
          $refAA[$i] = $codonMap->codon2aa($site->[$codonSequenceIdx]);
        }
        
        # For indels we don't store newAA or newCodon
      }

      $i++;
    }
  } else {
    # my $newAA;
    # my $refAA;
    my $alleleCodonSequence;

    SNP_LOOP: for my $site ($multiple ? @$siteData : $siteData) {
      $codonNum[$i] = $site->[$codonNumberIdx];
      $codonSeq[$i] = $site->[$codonSequenceIdx];

      if(!defined $site->[$codonSequenceIdx]) {
        $i++;
        next SNP_LOOP;
      }

      # We store as 0-based, users probably expect 1 based
      $codonPos[$i] = $site->[$codonPositionIdx] + 1;

      if(length($site->[$codonSequenceIdx]) != 3) {
        $i++;
        next SNP_LOOP;
      }

      #make a codon where the reference base is swapped for the allele
      $alleleCodonSequence = $site->[$codonSequenceIdx];

      # If codon is on the opposite strand, invert the allele
      # Note that $site->[$codonPositionIdx] MUST be 0-based for this to work
      if( $site->[$strandIdx] eq '-' ) {
        substr($alleleCodonSequence, $site->[$codonPositionIdx], 1) = $negativeStrandTranslation->{$allele};
      } else {
        substr($alleleCodonSequence, $site->[$codonPositionIdx], 1) = $allele;
      }

      $newCodon[$i] = $alleleCodonSequence;

      $newAA[$i] = $codonMap->codon2aa($alleleCodonSequence);
      $refAA[$i] = $codonMap->codon2aa($site->[$codonSequenceIdx]);

      if(!defined $newAA[$i]) {
        $i++;
        next SNP_LOOP;
      }

      if($refAA[$i] eq $newAA[$i]) {
        $funcAccum[$i] = $silent;
      } elsif($newAA[$i] eq '*') {
        $funcAccum[$i] = $stopGain;
      } elsif($refAA[$i] eq '*') {
        $funcAccum[$i] = $stopLoss;
      } elsif($codonNum[$i] == 1) {
        $funcAccum[$i] = $startLoss;
      } else {
        $funcAccum[$i] = $replacement;
      }

      $i++;
    }
  }

  if(!$multiple) {
    $out[$self->{_codonPosFidx}] = $codonPos[0];
    $out[$self->{_codonFIdx}] = $codonSeq[0];
    $out[$self->{_codonNumFidx}] = $codonNum[0];
    $out[$self->{_alleleFuncFidx}] = $funcAccum[0];
    $out[$self->{_refAaFidx}] = $refAA[0];
    $out[$self->{_altAaFidx}] = $newAA[0];
    $out[$self->{_altCodonFidx}] = $newCodon[0];
  } else {
    $out[$self->{_codonPosFidx}] = \@codonPos;
    $out[$self->{_codonFIdx}] = \@codonSeq;
    $out[$self->{_codonNumFidx}] = \@codonNum;
    $out[$self->{_alleleFuncFidx}] = \@funcAccum;
    $out[$self->{_refAaFidx}] = \@refAA;
    $out[$self->{_altAaFidx}] = \@newAA;
    $out[$self->{_altCodonFidx}] = \@newCodon;
  }

  return accumOut($alleleIdx, $positionIdx, $outAccum, \@out);
};

sub accumOut {
  # my ($alleleIdx, $positionIdx, $outAccum, $outAref) = @_;
  #     $_[0]     , $_[1]       , $_[2]    , $_[3]

  # for my $featureIdx (0 .. $#$outAref) {
  my $i = 0;
  for my $feature (@{$_[3]}) {
    #$outAccum->[$featureIdx][$alleleIdx][$positionIdx] = $outAref->[$featureIdx];
    $_[2]->[$i][$_[1]] = $feature;
    $i++;
  }

  #return $outAccum;
  return $_[2];
}
__PACKAGE__->meta->make_immutable;

1;