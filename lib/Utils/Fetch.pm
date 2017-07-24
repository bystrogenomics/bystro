use 5.10.0;
use strict;
use warnings;

package Utils::Fetch;

our $VERSION = '0.001';

# ABSTRACT: Fetch anything specified by remote_dir + remote_files 
# or sql_statement

use Mouse 2;

extends 'Utils::Base';

use namespace::autoclean;
use File::Which qw(which);
use Path::Tiny;
use YAML::XS qw/Dump/;

use Utils::SqlWriter;

use DDP;

# wget, ftp, whatever
# has fetch_program => (is => 'ro', writer => '_setFetchProgram');
# has fetch_program_arguments => (is => 'ro', writer => '_setFetchProgramArguments');
# has fetch_command => (is => 'ro');
has wget => (is => 'ro', init_arg => undef, writer => '_setWget');
has rsync => (is => 'ro', init_arg => undef, writer => '_setRsync');

sub BUILD {
  my $self = shift;

  if($self->_wantedTrack->{sql_statement}) {
    return;
  }

  if(!which('rsync') || !which('wget')) {
    $self->log('fatal', 'Fetch.pm requires rsync and wget when fetching remote_files');
  }

  $self->_setRsync(which('rsync'));
  $self->_setWget(which('wget'));
}

########################## The only public export  ######################
sub fetch {
  my $self = shift;

  if(defined $self->_wantedTrack->{remote_files} || defined $self->_wantedTrack->{remote_dir}) {
    return $self->_fetchFiles();
  }

  if(defined $self->_wantedTrack->{sql_statement}) {
    return $self->_fetchFromUCSCsql();
  }

  $self->log('fatal', "Couldn't find either remote_files + remote_dir,"
    . " or an sql_statement for this track");
}

########################## Main methods, which do the work  ######################
# These are called depending on whether sql_statement or remote_files + remote_dir given
sub _fetchFromUCSCsql {
  my $self = shift;
  
  my $sqlStatement = $self->_wantedTrack->{sql_statement};

  # What features are called according to our YAML config spec
  my $featuresKey = 'features';
  my $featuresIdx = index($sqlStatement, $featuresKey);

  if( $featuresIdx > -1 ) {
    if(! @{$self->_wantedTrack->{features}} ) {
      $self->log('fatal', "Requires features if sql_statement speciesi SELECT features")
    }

    my $trackFeatures;
    foreach(@{$self->_wantedTrack->{features}}) {
      # YAML config spec defines optional type on feature names, so some features
      # Can be hashes. Take only the feature name, ignore type, UCSC doesn't use them
      my $featureName;
      
      if(ref $_) {
        ($featureName) = %{$_};
      } else {
        $featureName = $_;
      }

      $trackFeatures .= $featureName . ',';
    }

    chop $trackFeatures;

    substr($sqlStatement, $featuresIdx, length($featuresKey) ) = $trackFeatures;
  }
  
  my $sqlWriter = Utils::SqlWriter->new({
    sql_statement => $sqlStatement,
    chromosomes => $self->_decodedConfig->{chromosomes},
    outputDir => $self->_localFilesDir,
    compress => 1,
  });

  # Returns the relative file names
  my @writtenFileNames = $sqlWriter->fetchAndWriteSQLData();

  $self->_wantedTrack->{local_files} = \@writtenFileNames;

  $self->_wantedTrack->{fetch_date} = $self->_dateOfRun;

  $self->_backupAndWriteConfig();

  $self->log('info', "Finished fetching data from sql");
}

sub _fetchFiles {
  my $self = shift;

  my $pathRe = qr/([a-z]+:\/\/)(\S+)/;
  my $remoteDir;
  my $remoteProtocol;

  my $fetchProgram;

  my $isRsync;

  if($self->_wantedTrack->{remote_dir}) {
    # remove http:// (or whatever protocol)
    $self->_wantedTrack->{remote_dir} =~ m/$pathRe/;

    if($1) {
      $isRsync = 0;
      $remoteProtocol = $1;
    } else {
      $isRsync = 0;
      $remoteProtocol = 'rsync://';
    }

    $remoteDir = $2;
  }

  my $outDir = $self->_localFilesDir;

  $self->_wantedTrack->{local_files} = [];

  for my $file ( @{$self->_wantedTrack->{remote_files}} ) {
    my $remoteUrl;

    if($remoteDir) {
      $remoteUrl = $remoteProtocol . path($remoteDir)->child($file)->stringify;
    } else {
      $file =~ m/$pathRe/;

      # This file is an absolute remote path
      if($1) {
        $remoteUrl = $file;
        $isRsync = 0;
      } else {
        $remoteUrl = "rsync://" . $2;
        $isRsync = 1;
      }
    }

    # Always outputs verbose, capture the arguments
    my $command;

    if($isRsync) {
      $command = $self->rsync . " -aPz $remoteUrl $outDir";
    } else {
      $command = $self->wget . " $remoteUrl -P $outDir";
    }

    $self->log('info', "Fetching: $command");

    # http://stackoverflow.com/questions/11514947/capture-the-output-of-perl-system
    open(my $fh, "-|", "$command") or $self->log('fatal', "Couldn't fork: $!\n");

    my $progress;
    while(<$fh>) {
      if($self->debug) { say $_ } # we may want to watch progress in stdout
      $self->log('info', $_);
    }
    close($fh);

    my $exitStatus = $?;

    if($exitStatus != 0) {
      $self->log('fatal', "Failed to fetch $file");
    }

    my $outFileName = $remoteDir ? $file : substr($file, rindex($file, '/') + 1);

    push @{ $self->_wantedTrack->{local_files} }, $outFileName;

    # stagger requests to be kind to the remote server
    sleep 3;
  }

  $self->_wantedTrack->{fetch_date} = $self->_dateOfRun;

  $self->_backupAndWriteConfig();

  $self->log('info', "Finished fetching all remote files");
}

__PACKAGE__->meta->make_immutable;

1;
