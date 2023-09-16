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
  This class not longer handles intergenic sites

=cut

use Mouse 2;

use namespace::autoclean;
use DDP;

use Seq::Tracks::Gene::Site;
use Seq::Tracks::Gene::Site::SiteTypeMap;
use Seq::Tracks::Gene::Site::CodonMap;
use Seq::Tracks::Gene::Definition;
use Seq::DBManager;
use Seq::Headers;

# Doesn't extend Seq::Tracks::Get to reduce inheritance depth, since most
# of that class is overriden anyhow (leaving only the headers property inheritance
# which isn't necessary since Seq::Headers is a singleton class)
extends 'Seq::Tracks::Base';

my $geneDef = Seq::Tracks::Gene::Definition->new();

### Set the features default that we get from the Gene track region database ###
has '+features' => (
  default => sub {
    return [ @{$geneDef->ucscGeneAref}, $geneDef->txErrorName ];
  },
);

with 'Seq::Tracks::Region::RegionTrackPath';

########### @Public attributes##########
########### Additional "features" that we will add to our output ##############
### Users may configure these; they are not stored, but computed ####

# These are features defined by Gene::Site, but we name them in Seq::Tracks::Gene
# Because it gets really confusing to track down the features defined in Seq::Tracks::Gene::Site
# TODO: rename these siteTypeField to match the interface used by Seq.pm (TODO: and Seq::Tracks::Sparse::Build)
has siteTypeKey      => ( is => 'ro', default => 'siteType' );
has strandKey        => ( is => 'ro', default => 'strand' );
has codonNumberKey   => ( is => 'ro', default => 'codonNumber' );
has codonPositionKey => ( is => 'ro', default => 'codonPosition' );

has codonSequenceKey        => ( is => 'ro', default => 'refCodon' );
has refAminoAcidKey         => ( is => 'ro', default => 'refAminoAcid' );
has newCodonKey             => ( is => 'ro', default => 'altCodon' );
has newAminoAcidKey         => ( is => 'ro', default => 'altAminoAcid' );
has exonicAlleleFunctionKey => ( is => 'ro', default => 'exonicAlleleFunction' );

# This is actually a stored value, added automatically
# has txErrorKey => (is => 'ro', default => 'txError');

# This is just the index corresponding to the transcript in the region db
# Not a real feature; we need to ask for "txNumberKey"
# In a future API, it may be worthwhile just storing a 'txNumber'
# property in the region database, to allow a user to just write 'txNumber'
# as a feature
# This is needed, because, as insane as it is, the transcript 'name'
# is not a unique key, and therefore is not a primary key, which makes it
# useless for lookup
has txNumberKey => ( is => 'ro', lazy => 1, default => 'txNumber');
has reportTxNumber => (is => 'ro', isa => 'Bool', default => 0);

# has hgvsPkey => (is => 'ro', default => 'hgvsP');
# has hgvsCkey => (is => 'ro', default => 'hgvsC');

########################## Private Attributes ##################################
########## The names of various features. These cannot be configured ##########
### Positions that aren't covered by a refSeq record are intergenic ###
### TODO: We don't output anything for these sites ###
### Because "intergenic" is just a label for missing , they are 100% equal ###

### txEffect possible values ###
# TODO: export these, and make them configurable
state $silent        = 'synonymous';
state $replacement   = 'nonSynonymous';
state $frameshift    = 'indel-frameshift';
state $inFrame       = 'indel-nonFrameshift';
state $indelBoundary = 'indel-exonBoundary';
state $startLoss     = 'startLoss';
state $stopLoss      = 'stopLoss';
state $stopGain      = 'stopGain';

has siteTypes => (
  is       => 'ro',
  init_arg => undef,
  lazy     => 1,
  default  => sub {
    return {
      replacement   => $replacement,
      silent        => $silent,
      frameshift    => $frameshift,
      inFrame       => $inFrame,
      indelBoundary => $indelBoundary,
      startLoss     => $startLoss,
      stopLoss      => $stopLoss,
      stopGain      => $stopGain,
    };
  }
);

# TODO: implement the truncated annotation
state $truncated = 'truncatedCodon';


### objects that get used by multiple subs, but shouldn't be public attributes ###
# All of these instantiated classes cannot be configured at instantiation time
# so safe to use in static context
state $siteUnpacker = Seq::Tracks::Gene::Site->new();
state $siteTypeMap  = Seq::Tracks::Gene::Site::SiteTypeMap->new();
state $codonMap     = Seq::Tracks::Gene::Site::CodonMap->new();

state $strandIdx        = $siteUnpacker->strandIdx;
state $siteTypeIdx      = $siteUnpacker->siteTypeIdx;
state $codonSequenceIdx = $siteUnpacker->codonSequenceIdx;
state $codonPositionIdx = $siteUnpacker->codonPositionIdx;
state $codonNumberIdx   = $siteUnpacker->codonNumberIdx;

state $negativeStrandTranslation = { A => 'T', C => 'G', G => 'C', T => 'A' };

#### Add our other "features", everything we find for this site ####
sub BUILD {
  my $self = shift;

  # Private variables, meant to cache often used data
  $self->{_allCachedDbNames} = {};
  $self->{_regionDb}         = {};

  $self->{_features} = $self->features;
  $self->{_dbName}   = $self->dbName;
  $self->{_db}       = Seq::DBManager->new();

  # Not including the txNumberKey;  this is separate from the annotations, which is
  # what these keys represent

  if ( $self->hasJoin ) {
    my $joinTrackName = $self->joinTrackName;

    $self->{_hasJoin} = 1;

    $self->{_flatJoinFeatures} =
    [ map { "$joinTrackName.$_" } @{ $self->joinTrackFeatures } ];

    my $i = 0;
    for my $fName ( @{ $self->joinTrackFeatures } ) {
      $self->{_allCachedDbNames}{ $self->{_flatJoinFeatures}[$i] } =
      $self->getFieldDbName($fName);
      $i++;
    }
  }

  for my $fName ( @{ $self->{_features} } ) {
    $self->{_allCachedDbNames}{$fName} = $self->getFieldDbName($fName);
  }
}

sub setHeaders {
  my $self     = shift;
  my @features = @{ $self->features };

  my $headers = Seq::Headers->new();

  ########################## Create the header #################################
  # don't mutate $self->features or flatJoinFeatures
  # those features should be keys in the region db
  # whereas siteType, exonicAlleleFunction, etc may be computed features
  # TODO: have clearer separation between computed and stored features
  ##############################################################################
  unshift @features, $self->siteTypeKey, $self->exonicAlleleFunctionKey,
  $self->codonSequenceKey, $self->newCodonKey, $self->refAminoAcidKey,
  $self->newAminoAcidKey, $self->codonPositionKey,
  $self->codonNumberKey, $self->strandKey;    #$self->hgvsCkey, $self->hgvsPkey

  if ( $self->{_flatJoinFeatures} ) {
    push @features, @{ $self->{_flatJoinFeatures} };
  }

  if($self->reportTxNumber) {
    push @features, $self->txNumberKey;
  }

  #  Prepend some custom features
  #  Providing 1 as the last argument means "prepend" instead of append
  #  So these features will come before any other refSeq.* features
  $headers->addFeaturesToHeader( \@features, $self->name );

  my @allGeneTrackFeatures = @{ $headers->getParentFeatures( $self->name ) };

  # Get the output index of each feature we added to the header
  # This is the index in this tracks output array
  # and includes features added to header, using addFeatureToHeader
  # such as the join track feature names
  # and siteType, strand, codonNumber, etc.
  for my $i ( 0 .. $#allGeneTrackFeatures ) {
    $self->{_featureIdxMap}{ $allGeneTrackFeatures[$i] } = $i;
  }

  ############### Store the output indices for computed features ###############
  # so that we can manually include them in the output
  my $lastFeatIdx = $#allGeneTrackFeatures;

  $self->{_featIdx}        = [ 0 .. $lastFeatIdx ];

  $self->{_strandFidx} = $self->{_featureIdxMap}{ $self->strandKey };
  $self->{_siteFidx}   = $self->{_featureIdxMap}{ $self->siteTypeKey };
  # Avoid accessor penalties by aliasing to the $self hash
  # These correspond to all of the sites held in Gene::Site
  $self->{_codonSidx}    = $self->{_featureIdxMap}{ $self->codonSequenceKey };
  $self->{_codonPosFidx} = $self->{_featureIdxMap}{ $self->codonPositionKey };
  $self->{_codonNumFidx} = $self->{_featureIdxMap}{ $self->codonNumberKey };

  # The values for these keys we calculate at get() time.
  $self->{_refAaFidx}      = $self->{_featureIdxMap}{ $self->refAminoAcidKey };
  $self->{_altCodonSidx}   = $self->{_featureIdxMap}{ $self->newCodonKey };
  $self->{_altAaFidx}      = $self->{_featureIdxMap}{ $self->newAminoAcidKey };
  $self->{_alleleFuncFidx} = $self->{_featureIdxMap}{ $self->exonicAlleleFunctionKey };

  $self->{_txNumberFidx} =  $self->{_featureIdxMap}{ $self->txNumberKey };

  # $self->{_hgvsCidx} = $self->{_featureIdxMap}{$self->hgvsCkey};
  # $self->{_hgvsPidx} = $self->{_featureIdxMap}{$self->hgvsPkey};
}

sub get {
  #my ($self, $dbData, $chr, $refBase, $alt, $posIdx, $outAccum) = @_;
  #    $_[0], $_[1], $_[1], $_[3],   $_[4],   $_[5]    $_[6]
  # WARNING: If $_[1]->[$_[0]->{_dbName} isn't defined, will be treated as the 0 index!!!
  # therefore return here if that is the case
  # ~1/2 of sites will have no gene track entry (including all non-coding, 2% coding)
  if ( !defined $_[1]->[ $_[0]->{_dbName} ] ) {
    for my $i ( @{ $_[0]->{_featIdx} } ) {
      $_[6]->[$i][ $_[5] ] = undef;
    }

    return $_[6];
  }

  my ( $self, $dbData, $chr, $ref, $alt, $posIdx, $outAccum ) = @_;

  # my @out;
  # # Set the out array to the size we need; undef for any indices we don't add here

  # Cached field names to make things easier to read
  my $cachedDbNames = $self->{_allCachedDbNames};
  my $idxMap        = $self->{_featureIdxMap};

  ################# Cache track's region data ##############
  # returns an array
  $self->{_regionDb}{$chr} //= $self->{_db}->dbReadAll( $self->regionTrackPath($chr) );

  my $geneDb = $self->{_regionDb}{$chr};

  ####### Get all transcript numbers, and site data for this position #########

  #<ArrayRef> $unpackedSites ; <ArrayRef|Int> $txNumbers
  my ( $siteData, $txNumbers, $multiple );

  #Reads:
  # ( $dbData->[$self->{_dbName}] ) {
  ( $txNumbers, $siteData ) = $siteUnpacker->unpack( $dbData->[ $self->{_dbName} ] );
  $multiple = ref $txNumbers ? $#$txNumbers : 0;

  if ( $self->{_hasJoin} ) {
    # For join tracks, use only the entry for the first of multiple transcripts
    # Because the data stored is always identical at one position
    my $num = $multiple ? $txNumbers->[0] : $txNumbers;
    # http://ideone.com/jlImGA
    for my $fName ( @{ $self->{_flatJoinFeatures} } ) {
      $outAccum->[ $idxMap->{$fName} ][$posIdx] =
      $geneDb->[$num]{ $cachedDbNames->{$fName} };
    }
  }

  # Needs to be done early, in case !hasCodon
  if(defined $self->{_txNumberFidx}) {
    $outAccum->[ $self->{_txNumberFidx} ][$posIdx] = $txNumbers;
  }

  ################## Populate site information ########################
  # save unpacked sites, for use in txEffectsKey population #####
  # moose attrs are very slow, cache
  # Push, because we'll use the indexes in calculating alleles
  # TODO: Better handling of truncated codons
  # Avoid a bunch of \;\ for non-coding sites
  # By not setting _codonNumberKey, _codonPositionKey, _codonSequenceKey if !hasCodon
  my $hasCodon;
  if ($multiple) {
    for my $site (@$siteData) {
      push @{ $outAccum->[ $self->{_strandFidx} ][$posIdx] }, $site->[$strandIdx];
      push @{ $outAccum->[ $self->{_siteFidx} ][$posIdx] },   $site->[$siteTypeIdx];

      if ( !$hasCodon && defined $site->[$codonSequenceIdx] ) {
        $hasCodon = 1;
      }
    }
  } else {
    # This is a bit messy; we should just push anyway, or be consistent
    # and never store an array unless multiple, through Gene.pm
    # Currently we rely on the output package to do the right thing
    $outAccum->[ $self->{_strandFidx} ][$posIdx] = $siteData->[$strandIdx];
    $outAccum->[ $self->{_siteFidx} ][$posIdx]   = $siteData->[$siteTypeIdx];

    if ( defined $siteData->[$codonSequenceIdx] ) {
      $hasCodon = 1;
    }
  }
  # ################# Populate geneTrack's user-defined features #####################
  #Reads:            $self->{_features}
  if ($multiple) {
    for my $feature ( @{ $self->{_features} } ) {
      $outAccum->[ $idxMap->{$feature} ][$posIdx] =
      [ map { $geneDb->[$_]{ $cachedDbNames->{$feature} } } @$txNumbers ];
    }
  } else {
    for my $feature ( @{ $self->{_features} } ) {
      $outAccum->[ $idxMap->{$feature} ][$posIdx] =
      $geneDb->[$txNumbers]{ $cachedDbNames->{$feature} };
    }
  }

  if ( !$hasCodon ) {
    $outAccum->[ $self->{_codonPosFidx} ][$posIdx]   = undef;
    $outAccum->[ $self->{_codonNumFidx} ][$posIdx]   = undef;
    $outAccum->[ $self->{_alleleFuncFidx} ][$posIdx] = undef;
    $outAccum->[ $self->{_refAaFidx} ][$posIdx]      = undef;
    $outAccum->[ $self->{_altAaFidx} ][$posIdx]      = undef;
    $outAccum->[ $self->{_codonSidx} ][$posIdx]      = undef;
    $outAccum->[ $self->{_altCodonSidx} ][$posIdx]   = undef;

    return;
  }

  ######Populate _codon*Key, exonicAlleleFunction, amion acids keys ############

  my ( $i, @funcAccum, @codonNum, @codonSeq, @codonPos, @refAA, @newAA, @newCodon );
  # Set undefs for every position, other than the ones we need
  # So that we don't need to push undef's to keep transcript order
  $#funcAccum = $#codonNum = $#codonSeq = $#codonPos = $#refAA = $#newAA = $#newCodon =
  $multiple;

  $i = 0;

  if ( length($alt) > 1 ) {
    # Indels get everything besides the _*AminoAcidKey and _newCodonKey
    my $indelAllele =
      substr( $alt, 0, 1 ) eq '+'
    ? length( substr( $alt, 1 ) ) % 3
      ? $frameshift
      : $inFrame
    : int($alt) % 3 ? $frameshift
    :                 $inFrame;

    for my $site ( $multiple ? @$siteData : $siteData ) {
      $codonNum[$i] = $site->[$codonNumberIdx];
      $codonSeq[$i] = $site->[$codonSequenceIdx];

      if ( defined $site->[$codonSequenceIdx] ) {
        $funcAccum[$i] = $indelAllele;

        # Codon position only exists (and always does) when codonSequence does
        # We store codonPosition as 0-based, users probably expect 1 based
        $codonPos[$i] = $site->[$codonPositionIdx] + 1;

        if ( length( $site->[$codonSequenceIdx] ) == 3 ) {
          $refAA[$i] = $codonMap->codon2aa( $site->[$codonSequenceIdx] );
        }

        # For indels we don't store newAA or newCodon
        # or hgvs notation
      }

      $i++;
    }
  } else {
    my $altCodonSequence;

    SNP_LOOP: for my $site ( $multiple ? @$siteData : $siteData ) {
      $codonNum[$i] = $site->[$codonNumberIdx];
      $codonSeq[$i] = $site->[$codonSequenceIdx];

      if ( !defined $site->[$codonSequenceIdx] ) {
        $i++;
        next SNP_LOOP;
      }

      # We store as 0-based, users probably expect 1 based
      $codonPos[$i] = $site->[$codonPositionIdx] + 1;

      if ( length( $site->[$codonSequenceIdx] ) != 3 ) {
        $i++;
        next SNP_LOOP;
      }

      #make a codon where the reference base is swapped for the allele
      $altCodonSequence = $site->[$codonSequenceIdx];

      # If codon is on the opposite strand, invert the allele
      # Note that $site->[$codonPositionIdx] MUST be 0-based for this to work
      if ( $site->[$strandIdx] eq '-' ) {
        substr( $altCodonSequence, $site->[$codonPositionIdx], 1 ) =
        $negativeStrandTranslation->{$alt};
      } else {
        substr( $altCodonSequence, $site->[$codonPositionIdx], 1 ) = $alt;
      }

      $newCodon[$i] = $altCodonSequence;

      $newAA[$i] = $codonMap->codon2aa($altCodonSequence);
      $refAA[$i] = $codonMap->codon2aa( $site->[$codonSequenceIdx] );

      if ( !defined $newAA[$i] ) {
        $i++;
        next SNP_LOOP;
      }

      if ( $refAA[$i] eq $newAA[$i] ) {
        $funcAccum[$i] = $silent;
      } elsif ( $newAA[$i] eq '*' ) {
        $funcAccum[$i] = $stopGain;
      } elsif ( $refAA[$i] eq '*' ) {
        $funcAccum[$i] = $stopLoss;
      } elsif ( $codonNum[$i] == 1 ) {
        $funcAccum[$i] = $startLoss;
      } else {
        $funcAccum[$i] = $replacement;
      }

      # $hgvsC[$i] = 'c.' . $ref . ($codonNum[$i] * 3 - (3 - $codonPos[$i])) . $alt;
      # $hgvsP[$i] = 'p.' . $refAA[$i] . $codonNum[$i] . $newAA[$i];

      $i++;
    }
  }

  if($multiple) {
    $outAccum->[ $self->{_codonPosFidx} ][$posIdx]   = \@codonPos;
    $outAccum->[ $self->{_codonNumFidx} ][$posIdx]   = \@codonNum;
    $outAccum->[ $self->{_alleleFuncFidx} ][$posIdx] = \@funcAccum;
    $outAccum->[ $self->{_refAaFidx} ][$posIdx]      = \@refAA;
    $outAccum->[ $self->{_altAaFidx} ][$posIdx]      = \@newAA;
    $outAccum->[ $self->{_codonSidx} ][$posIdx]      = \@codonSeq;
    $outAccum->[ $self->{_altCodonSidx} ][$posIdx]   = \@newCodon;

    return $outAccum;
  }

  $outAccum->[ $self->{_codonPosFidx} ][$posIdx]   = $codonPos[0];
  $outAccum->[ $self->{_codonNumFidx} ][$posIdx]   = $codonNum[0];
  $outAccum->[ $self->{_alleleFuncFidx} ][$posIdx] = $funcAccum[0];
  $outAccum->[ $self->{_refAaFidx} ][$posIdx]      = $refAA[0];
  $outAccum->[ $self->{_altAaFidx} ][$posIdx]      = $newAA[0];
  $outAccum->[ $self->{_codonSidx} ][$posIdx]      = $codonSeq[0];
  $outAccum->[ $self->{_altCodonSidx} ][$posIdx]   = $newCodon[0];

  # $outAccum->[$self->{_hgvsCidx}][$posIdx] = \@hgvsC;
  # $outAccum->[$self->{_hgvsPidx}][$posIdx] = \@hgvsP;
  return $outAccum;
}

__PACKAGE__->meta->make_immutable;

1;
