use 5.10.0;
use strict;
use warnings;

#Stores all 64 possible codons in a numerical map
#Stores all 64 possible codons mapped to single-leter amino-acids

# Safe for use when instantiated to static variable; no set - able properties
package Seq::Tracks::Gene::Site::CodonMap;
use DDP;
use Mouse 2;
use namespace::autoclean;

# Store both normal and truncated AA, to remove ambiguity for consumers
# 0 to 64 and 0 to 32 take 1 byte in msgpack
state $codonMap = {
  AAA => 1, AAC => 2, AAG => 3, AAT => 4, ACA => 5, ACC => 6, ACG => 7, ACT => 8,
  AGA => 9, AGC => 10, AGG => 11, AGT => 12, ATA => 13, ATC => 14, ATG => 15, ATT => 16,
  CAA => 17, CAC => 18, CAG => 19, CAT => 20, CCA => 21, CCC => 22, CCG => 23, CCT => 24, 
  CGA => 25, CGC => 26, CGG => 27, CGT => 28, CTA => 29, CTC => 30, CTG => 31, CTT => 32,
  GAA => 33, GAC => 34, GAG => 35, GAT => 36, GCA => 37, GCC => 38, GCG => 39, GCT => 40,
  GGA => 41, GGC => 42, GGG => 43, GGT => 44, GTA => 45, GTC => 46, GTG => 47, GTT => 48,
  TAA => 49, TAC => 50, TAG => 51, TAT => 52, TCA => 53, TCC => 54, TCG => 55, TCT => 56,
  TGA => 57, TGC => 58, TGG => 59, TGT => 60, TTA => 61, TTC => 62, TTG => 63, TTT => 64,

  #truncated
  AA => -1, AC => -2, AG => -3, AT => -4, CA => -5, CC => -6, CG => -7, CT => -8,
  GA => -9, GC => -10, GG => -11, GT => -12, TA => -13, TC => -14, TG => -15, TT => -16,
  A => -17, C => -18, G => -19, T => -20, 
};

sub codon2Num {
  #my ( $self, $codon ) = @_;
  #will return undefined if not found
  return $codonMap->{ $_[1] };
}

state $codonInverseMap = { map { $codonMap->{$_} => $_ } keys %$codonMap };

sub num2Codon {
  #my ( $self, $codon ) = @_;
  #will return undefined if not found
  
  return $codonInverseMap->{ $_[1] };
}

state $codonAAmap = {
  "AAA" => "K", "AAC" => "N", "AAG" => "K", "AAT" => "N",
  "ACA" => "T", "ACC" => "T", "ACG" => "T", "ACT" => "T",
  "AGA" => "R", "AGC" => "S", "AGG" => "R", "AGT" => "S",
  "ATA" => "I", "ATC" => "I", "ATG" => "M", "ATT" => "I",
  "CAA" => "Q", "CAC" => "H", "CAG" => "Q", "CAT" => "H",
  "CCA" => "P", "CCC" => "P", "CCG" => "P", "CCT" => "P",
  "CGA" => "R", "CGC" => "R", "CGG" => "R", "CGT" => "R",
  "CTA" => "L", "CTC" => "L", "CTG" => "L", "CTT" => "L",
  "GAA" => "E", "GAC" => "D", "GAG" => "E", "GAT" => "D",
  "GCA" => "A", "GCC" => "A", "GCG" => "A", "GCT" => "A",
  "GGA" => "G", "GGC" => "G", "GGG" => "G", "GGT" => "G",
  "GTA" => "V", "GTC" => "V", "GTG" => "V", "GTT" => "V",
  "TAA" => "*", "TAC" => "Y", "TAG" => "*", "TAT" => "Y",
  "TCA" => "S", "TCC" => "S", "TCG" => "S", "TCT" => "S",
  "TGA" => "*", "TGC" => "C", "TGG" => "W", "TGT" => "C",
  "TTA" => "L", "TTC" => "F", "TTG" => "L", "TTT" => "F"
};

sub codon2aa {
  #my ( $self, $codon ) = @_;
  #will return undefined if not found
  return $codonAAmap->{ $_[1] };
}

__PACKAGE__->meta->make_immutable;
1;