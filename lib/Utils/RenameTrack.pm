use 5.10.0;
use strict;
use warnings;

use lib '../';
# Takes a yaml file that defines one local file, and splits it on chromosome
# Only works for tab-delimitd files that have the c
package Utils::RenameTrack;

our $VERSION = '0.001';

use Mouse 2;
use namespace::autoclean;
use Path::Tiny qw/path/;

use DDP;

use Seq::Tracks::Build::LocalFilesPaths;
use Seq::Tracks::Base::MapTrackNames;
use List::MoreUtils qw/first_index/;

# _localFilesDir, _decodedConfig, compress, _wantedTrack, _setConfig, and logPath, 
extends 'Utils::Base';

########## Arguments accepted ##############
# Take the CADD file and make it a bed file
has renameTo => (is => 'ro', isa => 'Str', required => 1);
has dryRun => (is => 'ro', default => 0);

############# Private ############

sub BUILD {
  my $self = shift;

  my $databaseDir = $self->_decodedConfig->{database_dir};

  if(!$databaseDir) {
    $self->log('fatal', "database_dir required in config file for Utils::RenameTrack to work");
    return;
  }

  # DBManager acts as a singleton. It is configured once, and then consumed repeatedly
  # However, in long running processes, this can lead to misconfiguration issues
  # and worse, environments created in one process, then copied during forking, to others
  # To combat this, every time Seq::Base is called, we re-set/initialzied the static
  # properties that create this behavior
  Seq::DBManager::initialize();

  # Since we never have more than one database_dir, it's a global property we can set
  # in this package, which Seq.pm and Seq::Build extend from
  Seq::DBManager::setGlobalDatabaseDir($databaseDir);
}

sub go {
  my $self= shift;

  #TODO: rename dryRun to dryRun in main package
  my $trackNameMapper = Seq::Tracks::Base::MapTrackNames->new({dryRun => $self->dryRun});
    
  my $err = $trackNameMapper->renameTrack($self->name, $self->renameTo);

  if(!$err) {
    $self->log('info', "Renamed track from " . $self->name . " to " . $self->renameTo);
  } else {
    $self->log('info', "Failed to rename track " . $self->name . " because $err");
    return;
  }
  

  $self->_wantedTrack->{name} = $self->renameTo;

  #TODO: support renaming for the other fields
  if(defined $self->_decodedConfig->{statistics} && defined $self->_decodedConfig->{statistics}{dbSNPnameField}) {
    if($self->_decodedConfig->{statistics}{dbSNPnameField} eq $self->name) {
      $self->_decodedConfig->{statistics}{dbSNPnameField} = $self->renameTo;
    }
  }

  if(defined $self->_decodedConfig->{output} && defined $self->_decodedConfig->{output}{order}) {
    my $trackOrderIdx = first_index { $_ eq $self->name } @{ $self->_decodedConfig->{output}{order} };

    if( $trackOrderIdx > -1 ) {
      $self->_decodedConfig->{output}{order}[$trackOrderIdx] = $self->renameTo;
    }
  }

  my $metaPath = path($self->_decodedConfig->{database_dir})->child($self->name . '_meta');

  if(-e path($self->_decodedConfig->{database_dir})->child($self->name . '_meta') ) {
    $metaPath->move( path( $self->_decodedConfig->{database_dir} )->child($self->renameTo . '_meta') );
  }

  $self->_wantedTrack->{renameTrack_date} = $self->_dateOfRun;
  
  $self->_backupAndWriteConfig();
}

__PACKAGE__->meta->make_immutable;
1;
