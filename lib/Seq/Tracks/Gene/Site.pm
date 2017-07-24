use 5.10.0;
use strict;
use warnings;

#The goal of this is to both set and retrieve the information for a single
#position requested by the user
#So what is stored here is for the main database, and not 
#Breaking this thing down to fit in the new contxt
#based on Seq::Gene in (kyoto-based) seq branch
#except _get_gene_data moved to Seq::Tracks::GeneTrack::Build

# This package is meant for use in a static variable; it has no set - able 
# instance attributes

# TODO: Is Seq::Role::Message use safe in a threaded environment during building?
package Seq::Tracks::Gene::Site;

use Mouse 2;
use Scalar::Util qw/looks_like_number/;
use DDP;

use Seq::Tracks::Gene::Site::SiteTypeMap;
use Seq::Tracks::Gene::Site::CodonMap;

#exports log method to $self
with 'Seq::Role::Message';

#since the two Site:: packages are tightly coupled to packedCodon and 
#unpackCodon, I am making them public
#internally using the variables directly, because when called tens of millions
#of times, $self->codonMap may cost noticeable performance
#TODO: test that theory
#TODO: Remove these, consumers should just call Site::Class directly?
#Maybe not, because used to make combinedMap below
state $siteTypeMap = Seq::Tracks::Gene::Site::SiteTypeMap->new();
has siteTypeMap => (
  is => 'ro',
  init_arg => undef,
  lazy => 1,
  default => sub{ return $siteTypeMap },
);

state $codonMap = Seq::Tracks::Gene::Site::CodonMap->new();
has codonMap => (
  is => 'ro',
  init_arg => undef,
  lazy => 1,
  default => sub{ return $codonMap },
);

#These describe the site.
#Note that unpack will return both the transcript number and the site information
#So these indices refer only to the second value on, which represent
#The site information other than the transcript number (which is a reference to the region db)
#These must remain 0 - 4, used as constants in unpack
#They are simply exported for the use in consumers here
#Ex: 1-txSite : [txNubmer1, combinedStrandSitType, codonNumber, codonPosition, codonSequence]
#Expanded to: [txNumber1, strand, siteType, codonNumber, codonPosition, codonSequence] (in unpack)
has strandIdx => (is => 'ro', init_arg => undef, lazy => 1, default => 0);
has siteTypeIdx => (is => 'ro', init_arg => undef, lazy => 1, default => 1);
has codonNumberIdx => (is => 'ro', init_arg => undef, lazy => 1, default => 2);
has codonPositionIdx => (is => 'ro', init_arg => undef, lazy => 1, default => 3);
has codonSequenceIdx => (is => 'ro', init_arg => undef, lazy => 1, default => 4);

#pack strands as small integers, save a byte in messagepack
state $strandMap = { '-' => 0, '+' => 1, };

#combine strands with siteTypes #'-' will be a 0, '+' will be a 1;
state $combinedMap;
if(!$combinedMap) {
  foreach (keys %{$siteTypeMap->siteTypeMap}) {
    for my $num (0, 1) {
      $combinedMap->{$_ - $num} = [$num ? '+' : '-', $siteTypeMap->siteTypeMap->{$_}];
    }
  }
}

# Cost to pack an array using messagePack (which happens by default)
# Should be the same as the overhead for messagePack storing a string
# Unless the Perl messagePack implementation isn't good
# So store as array to save pack / unpack overhead
sub pack {
  my ($self, $txNumber, $siteType, $strand, $codonNumber, $codonPosition, $codonSeq) = @_;

  my @outArray;

  if( !defined $txNumber || !looks_like_number($txNumber) ) {
    $self->log('fatal', 'packCodon requires txNumber');
  }

  push @outArray, $txNumber;

  my $siteTypeNum = $siteTypeMap->getSiteTypeNum( $siteType );

  if(!defined $siteTypeNum) {
    $self->log('fatal', "site type $siteType not recognized");
  }

  if( ! defined $strandMap->{$strand} ) {
    $self->log('fatal', "Strand strand should be a + or -, got $strand");
  }

  #combines the strand and site type
  push @outArray, $siteTypeNum - $strandMap->{$strand};

  if(defined $codonNumber || defined $codonPosition || defined $codonSeq) {
    if(!defined $codonNumber && !defined $codonPosition && !defined $codonSeq) {
      $self->log('fatal', "Codons must be given codonNumber, codonPosition, and codonSeq"); 
    }

    if( !(looks_like_number( $codonPosition ) && looks_like_number( $codonNumber) ) ) {
      $self->log('fatal', "codonPosition && codonNumber must be numeric, got $codonPosition && $codonNumber");
    }

    push @outArray, $codonNumber;
    push @outArray, $codonPosition;

    my $codonSeqNumber  = $codonMap->codon2Num($codonSeq);

    if(length($codonSeq) != 3) {
      $self->log('debug', "codonSeqNumber for truncated is $codonSeqNumber");
    }

    #warning for now, this mimics the original codebase
    #TODO: do we want to store this as an error in the TX?
    if(!$codonSeqNumber) {
      $self->log('warn', "couldn\'t convert codon sequence $codonSeq to a number");
    } else {
      push @outArray, $codonSeqNumber;
    }
  }

  return \@outArray;
}
#@param <Seq::Tracks::Gene::Site> $self
#@param <ArrayRef> $codon
# This function assumes that the first value in any site array is the txNumber
# And the rest of values contain strandSiteTypeCombined, codonNumber, codonPosition, codonSequence
# The first value of that array is a combined siteType and strand
# Note also that we store codonPosition as 0 index (to try to store as 1/2 byte)
sub unpack {
  # my $self, $codon
  # $_[0],    $_[1]

  # Sites are stored in the form [ [ $txNumber1, codon1Val1, codon1Val2, ... codon1ValY ], [ txNumber2, ...], ...]
  # and for sites with only 1 transcript: # In the form [ $txNumber1, codon1Val1, codon1Val2,... ]
  # So if the first value isn't an array, we have a single transcript
  #! ref $codon->[0]
  if(!ref $_[1]->[0] ) {
    # If the length of our only codon is 2, which happens in intergenic cases
    # Then we return just the transcript number, and [strand, siteType]
    #   #@{$codon} == 2
    if( @{ $_[1] } == 2) {
      #returns: transcriptNum, [<Str>$strand, <Str>$siteType ]
      #      ( $codon->[0]),[( @{ $combinedMap->{ $codon->[1]} })  ]
      return ( $_[1]->[0], [ ( @{ $combinedMap->{ $_[1]->[1] } } ) ] );
    }
    
    # The first value in the return list is the transcript number
    #returns: transcriptNum, [<Str>$strand, <Str>$siteType ]
    #return ( $codon->[0]),[( @{ $combinedMap->{ $codon->[1]} }) , $codon->[$codonNumberIdx], $codon->[$codonPositionIdx] + 1,
    # $codonMap->num2Codon( $codon->[$codonSequenceIdx] ) ] );
    return ( $_[1]->[0], [ ( @{ $combinedMap->{ $_[1]->[1] } } ), $_[1]->[2], $_[1]->[3],
      $codonMap->num2Codon( $_[1]->[4] ) ] );
  }

  my (@site, @txNumbers);
  foreach (@{ $_[1] }) {
    # The first value is txNumber, and is always present
    push @txNumbers, $_->[0];

    if( @{$_} == 2) {
                # [ ( @{ $combinedMap->{ $_->[1] } } ) ]
      push @site, [ ( @{ $combinedMap->{ $_->[1] } } ) ];
      next;
    }
    #push @site,[ ( @{ $combinedMap->{ $_->[1] } } ), $_->[$codonNumberIdx], $_->[$codonPositionIdx] + 1,
    # $codonMap->num2Codon( $_->[$codonSequenceIdx] ) ];
    push @site, [ ( @{ $combinedMap->{ $_->[1] } } ), $_->[2], $_->[3],
      $codonMap->num2Codon( $_->[4] ) ];
  }

  return (\@txNumbers, \@site);
}

#Future API

# sub _unpackCodonBulk {
#   #my ($self, $codoAref) = @_;
#   #$codonStr == $_[1] 
#   #may be called a lot, so not using arg assignment
#   #Old version relied on pack/unpack, here are some informal tests:
#    #https://ideone.com/TFGjte
#     #https://ideone.com/dVy6WL
#     #my @unpackedCodon = $_[1] ? unpack('cAlcAAA', $_[1]) : (); 
#     #etc

#   for(my $i)

#   return {
#     $siteTypeKey => defined $_[1]->[0] ? $_[0]->getSiteTypeFromNum($_[1]->[0]) : undef,
#     $strandKey => $_[1]->[1],
#     $codonNumberKey => $_[1]->[2],
#     $codonPositionKey => $_[1]->[3],
#     $peptideKey => defined $_[1]->[4] ? $_[0]->codon2aa( $_[0]->num2Codon($_[1]->[4]) ) : undef
#   }
# }


# sub getCodonStrand {
#   return $unpackedCodonHref->{$_[0]->strandKey};
# }

# sub getCodonNumber {
#   return $unpackedCodonHref->{$_[0]->codonNumberKey};
# }

# sub getCodonPosition {
#   return $unpackedCodonHref->{$_[0]->codonPositionKey};
# }

# #https://ideone.com/cNQfwv
# sub getCodonSequence {
#   return $unpackedCodonHref->{$_[0]->codonSequenceKey};
# }

# sub getCodonAAresidue {
#   return $unpackedCodonHref->{$_[0]->peptideKey};
# }

# not in use yet
# sub hasCodon {
#   my ($self, $href) = @_;

#   return !!$href->{ $invFeatureMap->{refCodonSequence} };
# }

__PACKAGE__->meta->make_immutable;
1;