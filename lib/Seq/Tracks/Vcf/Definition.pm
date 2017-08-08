use 5.10.0;
use strict;
use warnings;

package Seq::Tracks::Vcf::Definition;
use Mouse::Role 2;
use DDP;

state $vcfFeatures = {CHROM => 1, POS => 1, ALT => 1, REF => 1, ID => 1, QUAL => 1, FILTER => 1};

# vcfFeature<CHROM|POS...> => name in "feature" list 
# Allows us to separate INFO features from first few columns
# TODO: Maybe switch to using indices always, rather than dbName system
has headerFeatures => (is => 'rw', init_arg => undef, isa => 'HashRef');

# Figure out which fields are from the columns preceeding INFO
# And store them in the headerFeatures
# Cannot use AROUND BUILDARGS; will prevent other buildargs from running
# Due to our complex inheritance pattern, and Mouse limitations vs Moose
before 'BUILD' => sub {
  my $self = shift;

  my $features = $self->features;

  if(!$features) {
    die "VCF build requires INFO features";
  }

  my %featuresMap;
  for(my $i = 0; $i < @{$features}; $i++) {
    $featuresMap{lc($features->[$i])} = $i;
  }

  my %fieldMap = map{ lc($_) => $self->fieldMap->{$_} } keys %{$self->fieldMap};

  my %headerFeatures;

  for my $vcfFeature (keys %$vcfFeatures) {
    my $idx = -9;

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

    if($idx == -9) {
      next;
    }

    $headerFeatures{$vcfFeature} = $self->features->[$idx];
  }

  # required
  if(!defined $headerFeatures{ALT}) {
    unshift @{$features}, 'alt';

    $headerFeatures{ALT} = 'alt';
  }

  $self->headerFeatures(\%headerFeatures);
};

no Mouse::Role;
1;