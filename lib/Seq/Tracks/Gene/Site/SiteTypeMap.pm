use 5.10.0;
use strict;
use warnings;

package Seq::Tracks::Gene::Site::SiteTypeMap;

use Mouse 2;
use Mouse::Util::TypeConstraints;
use DDP;
# Define allowable types

# Safe for use when instantiated to static variable; no set - able properties
state $codingSite = 'exonic';
has codingSiteType => (is=> 'ro', lazy => 1, init_arg => undef, default => sub{$codingSite} );
state $fivePrimeSite = 'UTR5';
has fivePrimeSiteType => (is=> 'ro', lazy => 1, init_arg => undef, default => sub{$fivePrimeSite} );
state $threePrimeSite = 'UTR3';
has threePrimeSiteType => (is=> 'ro', lazy => 1, init_arg => undef, default => sub{$threePrimeSite} );
state $spliceAcSite = 'spliceAcceptor';
has spliceAcSiteType => (is=> 'ro', lazy => 1, init_arg => undef, default => sub{$spliceAcSite} );
state $spliceDonSite = 'spliceDonor';
has spliceDonSiteType => (is=> 'ro', lazy => 1, init_arg => undef, default => sub{$spliceDonSite} );
state $ncRNAsite = 'ncRNA';
has ncRNAsiteType => (is=> 'ro', lazy => 1, init_arg => undef, default => sub{$ncRNAsite} );
state $intronicSite = 'intronic';
has intronicSiteType => (is=> 'ro', lazy => 1, init_arg => undef, default => sub{$intronicSite} );

# #Coding type always first; order of interest
state $siteTypes = [$codingSite, $fivePrimeSite, $threePrimeSite,
  $spliceAcSite, $spliceDonSite, $ncRNAsite, $intronicSite];

# #public
has siteTypes => (
  is => 'ro',
  isa => 'ArrayRef',
  traits => ['Array'],
  handles => {
    allSiteTypes => 'elements',
    getSiteType => 'get',
  },
  lazy => 1,
  init_arg => undef,
  default => sub{$siteTypes},
);

has nonCodingBase => (
  is => 'ro',
  isa => 'Int',
  init_arg => undef,
  lazy => 1,
  default => 1,
);

has codingBase => (
  is => 'ro',
  isa => 'Int',
  init_arg => undef,
  lazy => 1,
  default => 3,
);

has fivePrimeBase => (
  is => 'ro',
  isa => 'Int',
  init_arg => undef,
  lazy => 1,
  default => 5,
);

has threePrimeBase => (
  is => 'ro',
  isa => 'Int',
  init_arg => undef,
  lazy => 1,
  default => 7,
);

has spliceAcBase => (
  is => 'ro',
  isa => 'Int',
  init_arg => undef,
  lazy => 1,
  default => 9,
);

has spliceDonBase => (
  is => 'ro',
  isa => 'Int',
  init_arg => undef,
  lazy => 1,
  default => 11,
);

has intronicBase => (
  is => 'ro',
  isa => 'Int',
  init_arg => undef,
  lazy => 1,
  default => 13,
);

#TODO: should constrain values to GeneSiteType
has siteTypeMap => (
  is => 'ro',
  isa => 'HashRef',
  traits => ['Hash'],
  handles => {
    getSiteTypeFromNum => 'get',
  },
  lazy => 1,
  init_arg => undef,
  builder => '_buildSiteTypeMap',
);

sub _buildSiteTypeMap {
  my $self = shift;

  state $mapHref = {
    $self->nonCodingBase => $ncRNAsite,
    $self->codingBase => $codingSite,
    $self->fivePrimeBase => $fivePrimeSite,
    $self->threePrimeBase => $threePrimeSite,
    $self->spliceAcBase => $spliceAcSite,
    $self->spliceDonBase => $spliceDonSite,
    $self->intronicBase => $intronicSite,
  };

  return $mapHref;
}

#takes a GeneSite value and returns a number, matching the _siteTypeMap key
has siteTypeMapInverse => (
  is => 'ro',
  isa => 'HashRef',
  traits => ['Hash'],
  handles => {
    getSiteTypeNum => 'get',
  },
  lazy => 1,
  init_arg => undef,
  builder => '_buildSiteTypeMapInverse',
);

sub _buildSiteTypeMapInverse {
  my $self = shift;

  state $inverse =  { map { $self->siteTypeMap->{$_} => $_ } keys %{$self->siteTypeMap} };

  return $inverse;
}

has exonicSites => (
  is => 'ro',
  init_arg => undef,
  lazy => 1,
  isa => 'HashRef',
  traits => ['Hash'],
  handles => {
    isExonicSite => 'exists',
  },
  default => sub {
    return { map { $_ => 1 } ($codingSite, $ncRNAsite, $fivePrimeSite, $threePrimeSite) };
  },
);

__PACKAGE__->meta->make_immutable;
1;
