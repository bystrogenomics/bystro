use 5.10.0;
use strict;
use warnings;

package Utils::SqlWriter::Connection;
use DBI;

our $VERSION = '0.001';

# ABSTRACT: Fetch and write some data using UCSC's public SQL db
use Mouse 2;

use namespace::autoclean;

has driver  => ( is => 'ro', isa => 'Str',  default => "DBI:mysql" );
has host => ( is => 'ro', isa => 'Str', default  => "genome-mysql.cse.ucsc.edu");
has user => ( is => 'ro', isa => 'Str', default => "genome" );
has password => ( is => 'ro', isa => 'Str', );
has port     => ( is => 'ro', isa => 'Int', );
has socket   => ( is => 'ro', isa => 'Str', );

=method @public sub connect

  Build database object, and return a handle object

Called in: none

@params:

@return {DBI}
  A connection object

=cut

sub connect {
  my $self = shift;
  my $databaseName = shift;

  my $connection  = $self->driver;
  $connection .= ":database=$databaseName;host=" . $self->host if $self->host;
  $connection .= ";port=" . $self->port if $self->port;
  $connection .= ";mysql_socket=" . $self->port_num if $self->socket;
  $connection .= ";mysql_read_default_group=client";

  return DBI->connect( $connection, $self->user, $self->password, {
    RaiseError => 1, PrintError => 1, AutoCommit => 1
  } );
}

__PACKAGE__->meta->make_immutable();
1;