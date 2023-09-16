use 5.10.0;
use strict;
use warnings;

package Utils::Fetch;

our $VERSION = '0.001';

# ABSTRACT: Fetch anything specified by remoteDir . / . remoteFiles
# or an sql statement

use Mouse 2;

# Exports: _localFilesDir, _decodedConfig, compress, _wantedTrack, _setConfig, logPath, use_absolute_path
extends 'Utils::Base';

use namespace::autoclean;
use File::Which qw(which);
use Path::Tiny;
use YAML::XS qw/Dump/;

use Utils::SqlWriter;

use DDP;

# The sql connection config
has sql         => ( is => 'ro', isa => 'Maybe[Str]' );
has remoteFiles => ( is => 'ro', isa => 'Maybe[ArrayRef]' );
has remoteDir   => ( is => 'ro', isa => 'Maybe[Str]' );

has connection => ( is => 'ro', isa => 'Maybe[HashRef]' );

# Choose whether to use wget or rsync program to fetch
has aws   => ( is => 'ro', init_arg => undef, writer => '_setAws' );
has wget  => ( is => 'ro', init_arg => undef, writer => '_setWget' );
has rsync => ( is => 'ro', init_arg => undef, writer => '_setRsync' );

sub BUILD {
  my $self = shift;

  if ( defined $self->sql ) {
    return;
  }

  my $aws   = which('aws');
  my $wget  = which('wget');
  my $rsync = which('rsync');

  if ( !$rsync || !$wget ) {
    $self->log( 'fatal', 'Fetch.pm requires rsync and wget when fetching remoteFiles' );
  }

  $self->_setAws($aws);
  $self->_setRsync($rsync);
  $self->_setWget($wget);
}

########################## The only public export  ######################
sub go {
  my $self = shift;

  if ( defined $self->remoteFiles || defined $self->remoteDir ) {
    return $self->_fetchFiles();
  }

  if ( defined $self->sql ) {
    return $self->_fetchFromUCSCsql();
  }

  $self->log( 'fatal',
        "Couldn't find either remoteFiles + remoteDir,"
      . " or an sql statement for this track "
      . $self->name );
}

########################## Main methods, which do the work  ######################
# These are called depending on whether sql_statement or remoteFiles + remoteDir given
sub _fetchFromUCSCsql {
  my $self = shift;

  my $sqlStatement = $self->sql;

  # What features are called according to our YAML config spec
  my $featuresKey = '%features%';
  my $featuresIdx = index( $sqlStatement, $featuresKey );

  if ( $featuresIdx > -1 ) {
    if ( !@{ $self->_wantedTrack->{features} } ) {
      $self->log( 'fatal',
        "Requires features if sql_statement speciesi SELECT %features%" );
    }

    my $trackFeatures;
    foreach ( @{ $self->_wantedTrack->{features} } ) {

      # YAML config spec defines optional type on feature names, so some features
      # Can be hashes. Take only the feature name, ignore type, UCSC doesn't use them
      my $featureName;

      if ( ref $_ ) {
        ($featureName) = %{$_};
      }
      else {
        $featureName = $_;
      }

      $trackFeatures .= $featureName . ',';
    }

    chop $trackFeatures;

    substr( $sqlStatement, $featuresIdx, length($featuresKey) ) = $trackFeatures;
  }

  my $config = {
    sql         => $sqlStatement,
    assembly    => $self->_decodedConfig->{assembly},
    chromosomes => $self->_decodedConfig->{chromosomes},
    outputDir   => $self->_localFilesDir,
    compress    => 1,
  };

  if ( defined $self->connection ) {
    $config->{connection} = $self->connection;
  }

  my $sqlWriter = Utils::SqlWriter->new($config);

  # Returns the relative file names
  my @writtenFileNames = $sqlWriter->go();

  $self->_wantedTrack->{local_files} = \@writtenFileNames;

  $self->_backupAndWriteConfig();

  $self->log( 'info', "Finished fetching data from sql" );
}

sub _fetchFiles {
  my $self = shift;

  my $pathRe = qr/([a-z]+:\/\/)(\S+)/;
  my $remoteDir;
  my $remoteProtocol;

  my $fetchProgram;

  my $isRsync = 0;
  my $isS3    = 0;

  if ( $self->remoteDir ) {

    # remove http:// (or whatever protocol)
    $self->remoteDir =~ m/$pathRe/;

    if ($1) {
      $remoteProtocol = $1;
    }
    elsif ( $self->remoteDir =~ 's3://' ) {
      $isS3           = 1;
      $remoteProtocol = 's3://';
    }
    else {
      $isRsync        = 1;
      $remoteProtocol = 'rsync://';
    }

    $remoteDir = $2;
  }

  my $outDir = $self->_localFilesDir;

  $self->_wantedTrack->{local_files} = [];

  for my $file ( @{ $self->remoteFiles } ) {
    my $remoteUrl;

    if ($remoteDir) {
      $remoteUrl = $remoteProtocol . path($remoteDir)->child($file)->stringify;
    }
    else {
      $file =~ m/$pathRe/;

      # This file is an absolute remote path
      if ($1) {
        $remoteUrl = $file;
      }
      elsif ( $file =~ 's3://' ) {
        $remoteUrl = $file;
        $isS3      = 1;
      }
      else {
        $remoteUrl = "rsync://" . $2;
        $isRsync   = 1;
      }
    }

    # Always outputs verbose, capture the arguments
    my $command;

    if ($isRsync) {
      $command = $self->rsync . " -avPz $remoteUrl $outDir";
    }
    elsif ($isS3) {
      if ( !$self->aws ) {
        $self->log( 'fatal',
          "You requested an s3 remote file ($remoteUrl), but have no aws s3 cli installed. Please visit: https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-install.html"
        );
        return;
      }
      $command = $self->aws . " s3 cp $remoteUrl $outDir";
    }
    else {
      # -N option will clobber only if remote file is newer than local copy
      # -S preserves timestamp
      $command = $self->wget . " -N -S $remoteUrl -P $outDir";
    }

    $self->log( 'info', "Fetching: $command" );

    # http://stackoverflow.com/questions/11514947/capture-the-output-of-perl-system
    open( my $fh, "-|", "$command" )
      or $self->log( 'fatal', "Couldn't fork: $!\n" );

    my $progress;
    while (<$fh>) {
      if ( $self->debug ) {
        say $_;
      } # we may want to watch progress in stdout
      $self->log( 'info', $_ );
    }
    close($fh);

    my $exitStatus = $?;

    if ( $exitStatus != 0 ) {
      $self->log( 'fatal', "Failed to fetch $file" );
    }

    my $outFileName = $remoteDir ? $file : substr( $file, rindex( $file, '/' ) + 1 );

    push @{ $self->_wantedTrack->{local_files} }, $outFileName;

    # stagger requests to be kind to the remote server
    sleep 3;
  }

  $self->_backupAndWriteConfig();

  $self->log( 'info', "Finished fetching all remote files" );
}

__PACKAGE__->meta->make_immutable;

1;
