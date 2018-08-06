use 5.10.0;
use strict;
use warnings;

package Seq::Output::Delimiters;
use Mouse 2;
with 'Seq::Role::Message';

has valueDelimiter => (is => 'ro', isa => 'Str', default => ';');

has positionDelimiter => (is => 'ro', isa => 'Str', default => '|');

# Not currently used; one day may use again; this is the most inconvenient delimiter
# Could choose another at a later time
# Need \\; so need to escape twice
has alleleDelimiter => (is => 'ro', isa => 'Str',  default => "\\\\");

# Allows 1:n (or n:m) relationships between features of a single track
# Typically occurs with dbSNP (1 rs# for at least 2 alleles; in cases of 2 rs#
# need rs1;rs2 allele1_a/allele1_b;allele2_a/allele2_b to keep the order
# of rs1 => [allele1_a, allele1_b] rs2 => [allele2_a,allele2_b]
# So in short is expected to be used for the 3rd dimension of a 3D array (3-tensor)
# [ [ [overlapDelimiter] ] ]
has overlapDelimiter => (is => 'ro', isa => 'Str',  default => '/');

has fieldSeparator => (is => 'ro', isa => 'Str', default => "\t");

has emptyFieldChar => (is => 'ro',  isa => 'Str', default => '!');

has _makeCleanFunc => (is => 'ro', init_arg => undef, lazy => 1, default => sub {
  my $self = shift;

  my $vD = $self->valueDelimiter;
  my $pD = $self->positionDelimiter;
  my $aD = $self->alleleDelimiter;
  my $oD = $self->overlapDelimiter;

  my $re = qr/[$vD$pD$aD$oD]+/;

  return sub {
    #my ($line) = @_;
    #    $_[0]

    # modified $line ($_[0]) directly
    #/s modifier to squash multiple
    $_[0] =~ s/$re/ /gs;
  }
});

# Memoized cleaning function
has cleanDelims => (is => 'ro', init_arg => undef, lazy => 1, default => sub {
  my $self = shift;

  return $self->_makeCleanFunc();
});

__PACKAGE__->meta->make_immutable();
1;