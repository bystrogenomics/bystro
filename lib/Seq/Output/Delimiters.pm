use 5.10.0;
use strict;
use warnings;

package Seq::Output::Delimiters;
use Mouse 2;
use DDP;
with 'Seq::Role::Message';

has valueDelimiter => (is => 'ro', isa => 'Str', default => ';');

has positionDelimiter => (is => 'ro', isa => 'Str', default => '|');

# Not currently used; one day may use again; this is the most inconvenient delimiter
# Could choose another at a later time
# Need \\; so need to escape twice
# has alleleDelimiter => (is => 'ro', isa => 'Str',  default => '/');

# Allows 1:n (or n:m) relationships between features of a single track
# Typically occurs with dbSNP (1 rs# for at least 2 alleles; in cases of 2 rs#
# need rs1;rs2 allele1_a/allele1_b;allele2_a/allele2_b to keep the order
# of rs1 => [allele1_a, allele1_b] rs2 => [allele2_a,allele2_b]
# So in short is expected to be used for the 3rd dimension of a 3D array (3-tensor)
# [ [ [overlapDelimiter] ] ]
# Using \ because backslash is, at least in our data very infrequently used
# Regex requires double escaping of \; need to supply \\\\, which returns \\
has overlapDelimiter => (is => 'ro', isa => 'Str',  default => "\\\\");

has fieldSeparator => (is => 'ro', isa => 'Str', default => "\t");

has emptyFieldChar => (is => 'ro',  isa => 'Str', default => '!');

# What to replace the flagged characters with if found in a string
has globalReplaceChar => (is => 'ro', isa => 'Str', default => ',');

# Memoized cleaning function
has cleanDelims => (is => 'ro', init_arg => undef, lazy => 1, default => sub {
  my $self = shift;

  # my $aD = $self->alleleDelimiter;
  my $vD = $self->valueDelimiter;
  my $pD = $self->positionDelimiter;
  my $oD = $self->overlapDelimiter;
  my $gr = $self->globalReplaceChar;

  my $re = qr/[$vD$pD$oD]+/;
  my $reEnd = qr/[$vD$pD$oD$gr]+$/;

  return sub {
    #my ($line) = @_;
    #    $_[0]

    # modified $line ($_[0]) directly
    #/s modifier to squash multiple
    $_[0] =~ s/$re/$gr/gs;
    $_[0] =~ s/$reEnd//gs;
  }
});

has splitByField => (is => 'ro', init_arg => undef, lazy => 1, default => sub {
  my $self = shift;

  my $d = $self->fieldSeparator;

  my $re = qr/[$d]/;

  # Returns unmodified value, or a list of split values
  # Used in list context either 1 or many values emitted
  return sub {
    #my ($line) = @_;
    #    $_[0]

    # modified $line ($_[0]) directly
    #/s modifier to squash multiple
    return split /$re/s, $_[0];
  }
});

has splitByPosition => (is => 'ro', init_arg => undef, lazy => 1, default => sub {
  my $self = shift;

  my $d = $self->positionDelimiter;

  my $re = qr/[$d]/;

  # Returns unmodified value, or a list of split values
  # Used in list context either 1 or many values emitted
  return sub {
    #my ($line) = @_;
    #    $_[0]

    if(index($_[0], $d) == -1 ){
      return $_[0];
    }

    # modified $line ($_[0]) directly
    #/s modifier to squash multiple
    return split /$re/s, $_[0];
  }
});

has splitByOverlap => (is => 'ro', init_arg => undef, lazy => 1, default => sub {
  my $self = shift;

  my $d = $self->overlapDelimiter;

  my $re = qr/[$d]/;

  # Returns unmodified value, or a list of split values
  # Used in list context either 1 or many values emitted
  return sub {
    #my ($line) = @_;
    #    $_[0]

    if(index($_[0], $d) == -1 ){
      return $_[0];
    }

    # modified $line ($_[0]) directly
    #/s modifier to squash multiple
    return split /$re/s, $_[0];
  }
});

has splitByValue => (is => 'ro', init_arg => undef, lazy => 1, default => sub {
  my $self = shift;

  my $d = $self->valueDelimiter;

  my $re = qr/[$d]/;

  # Returns unmodified value, or a list of split values
  # Used in list context either 1 or many values emitted
  return sub {
    #my ($line) = @_;
    #    $_[0]

    if(index($_[0], $d) == -1 ){
      return $_[0];
    }

    # modified $line ($_[0]) directly
    #/s modifier to squash multiple
    return split /$re/s, $_[0];
  }
});

# has splitByAllele => (is => 'ro', init_arg => undef, lazy => 1, default => sub {
#   my $self = shift;

#   my $d = $self->alleleDelimiter;

#   my $re = qr/[$d]/;

#   # Returns unmodified value, or a list of split values
#   # Used in list context either 1 or many values emitted
#   return sub {
#     #my ($line) = @_;
#     #    $_[0]

#     if(index($_[0], $d) == -1 ){
#       return $_[0];
#     }

#     # modified $line ($_[0]) directly
#     #/s modifier to squash multiple
#     return split /$re/s, $_[0];
#   }
# });

__PACKAGE__->meta->make_immutable();
1;