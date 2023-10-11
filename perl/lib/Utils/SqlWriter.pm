use 5.10.0;
use strict;
use warnings;

package Utils::SqlWriter;

our $VERSION = '0.001';

# ABSTRACT: Fetch and write some data using UCSC's public SQL db
use Mouse 2;

use namespace::autoclean;
use Time::localtime;
use Path::Tiny qw/path/;

use Utils::SqlWriter::Connection;
with 'Seq::Role::IO', 'Seq::Role::Message';

# @param <Str> sql_statement : Valid SQL with fully qualified field names
has sql => ( is => 'ro', isa => 'Str', required => 1 );

# Where any downloaded or created files should be saved
has outputDir => ( is => 'ro', isa => 'Str', required => 1 );

has connection => ( is => 'ro', isa => 'Maybe[HashRef]' );

# @param <ArrayRef> chromosomes : All wanted chromosomes
has chromosomes => ( is => 'ro', isa => 'ArrayRef' );

# Compress the output?
has compress => ( is => 'ro', isa => 'Bool' );

has sqlClient => ( is => 'ro', init_arg => undef, writer => '_setSqlClient' );
######################### DB Configuartion Vars #########################
my $year         = localtime->year() + 1900;
my $mos          = localtime->mon() + 1;
my $day          = localtime->mday;
my $nowTimestamp = sprintf( "%d-%02d-%02d", $year, $mos, $day );

sub BUILD {
  my $self = shift;

  my $config = defined $self->connection ? { connection => $self->connection } : {};

  $self->_setSqlClient( Utils::SqlWriter::Connection->new($config) );
}

=method @public sub fetchAndWriteSQLData

  Read the SQL data and write to file

@return {DBI}
  A connection object

=cut

sub go {
  my $self = shift;

  my $extension = $self->compress ? 'gz' : 'txt';

  my $query = $self->sql;

  my $perChromosome;

  if ( $query =~ /\%chromosomes\%/ ) {
    $perChromosome = 1;
  }

  # We'll return the relative path to the files we wrote
  my @outRelativePaths;
  CHR_LOOP: for my $chr ( $perChromosome ? @{ $self->chromosomes } : 'fetch' ) {
    # for return data
    my @sql_data = ();

    my $query = $self->sql;

    ########### Restrict SQL fetching to just this chromosome ##############

    # Get the FQ table name (i.e hg19.refSeq instead of refSeq), to avoid
    if ($perChromosome) {
      $query =~ s/\%chromosomes\%/'$chr'/g;
    }

    # Will choose the first FROM; in complex
    $query =~ m/FROM\s(\S+)/i;

    ##### use database ######
    # If given database in connection object, use that, else try to infer
    my $databaseName;

    my $tableName = $1;

    # Check if table name is database.table
    if ( $tableName =~ /\S+\.\S+/ ) {
      ( $databaseName, $tableName ) = ( split( /\./, $tableName ) );
    }

    if ( defined $self->sqlClient->database ) {
      $databaseName = $self->sqlClient->database;
    }

    if ( !$databaseName ) {
      $self->log( 'fatal',
        "No database found: use a fully qualified table (database.table) or set the 'database' property in 'connection'"
      );
    }

    $self->log( 'info', "Set database name to $databaseName\n" );

    my $fileName = join '.', $databaseName, $tableName, $chr, $extension;

    $self->log( 'info', "Set file name to $fileName\n" );

    my $timestampName = join '.', $nowTimestamp, $fileName;

    # Save the fetched data to a timestamped file, then symlink it to a non-timestamped one
    # This allows non-destructive fetching
    my $symlinkedFile = path( $self->outputDir )->child($fileName)->absolute->stringify;
    my $targetFile =
      path( $self->outputDir )->child($timestampName)->absolute->stringify;

    # prepare file handle
    my $outFh = $self->getWriteFh($targetFile);

    $self->log( 'info', "Fetching from $databaseName: $query\n\n" );
    ########### Connect to database ##################
    my $dbh = $self->sqlClient->connect($databaseName);
    ########### Prepare and execute SQL ##############
    my $sth = $dbh->prepare($query) or $self->log( 'fatal', $dbh->errstr );

    $sth->execute or $self->log( 'fatal', $dbh->errstr );

    ########### Retrieve data ##############
    my $count = -1;
    while ( my @row = $sth->fetchrow_array ) {
      $count++;

      if ( $count == 0 ) {
        # Write header
        # Cleaner here, because there is nothing in {NAME} when empty query
        my @stuff = @{ $sth->{NAME} };
        push @sql_data, $sth->{NAME};
      }

      my $clean_row_aref = $self->_cleanRow( \@row );
      push @sql_data, $clean_row_aref;

      if ( @sql_data > 1000 ) {
        say $outFh join( "\n", map { join( "\t", @$_ ) } @sql_data );
        @sql_data = ();
      }
    }

    # leftovers
    if (@sql_data) {
      say $outFh join( "\n", map { join( "\t", @$_ ) } @sql_data );
      @sql_data = ();
    }

    $sth->finish();
    # Must commit before this works, or will get DESTROY before explicit disconnect()
    $dbh->disconnect();

    # We may not have data for all chromsoomes
    if ( $count > -1 ) {
      $self->log( "info", "Finished writing $targetFile\n\n" );

      if ( system("ln -s -f $targetFile $symlinkedFile") != 0 ) {
        $self->log( 'fatal', "Failed to symlink $targetFile -> $symlinkedFile\n\n" );
      }

      $self->log( 'info', "Symlinked $targetFile -> $symlinkedFile\n\n" );

      push @outRelativePaths, $fileName;
      next CHR_LOOP;
    }

    $self->log( "error",
      "No results found for $chr: \n query: $query, \n archive: $targetFile, \n output: $symlinkedFile)\n\n"
    );
    # # We may have had 0 results;
    # if (-z $targetFile) {
    #   unlink $targetFile;
    # }

    # if (-z $symlinkedFile) {
    #   unlink $symlinkedFile;
    # }

    # Throttle connection
    sleep 5;
  }

  return @outRelativePaths;
}

sub _cleanRow {
  my ( $self, $aref ) = @_;

  # http://stackoverflow.com/questions/2059817/why-is-perl-foreach-variable-assignment-modifying-the-values-in-the-array
  for my $ele (@$aref) {
    if ( !defined($ele) || $ele eq "" ) {
      $ele = "NA";
    }
  }

  return $aref;
}

__PACKAGE__->meta->make_immutable;

1;
