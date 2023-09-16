use 5.10.0;
use strict;
use warnings;

package Seq::Tracks::Build::LocalFilesPaths;

use Mouse 2;
use DDP;
use Path::Tiny qw/path/;
use File::Glob ':bsd_glob';

sub makeAbsolutePaths {
  my ($self, $filesDir, $trackName, $localFilesAref) = @_;
  
  my @localFiles;
  for my $localFile (@$localFilesAref) {
    if(path($localFile)->is_absolute) {
      push @localFiles, bsd_glob( $localFile );
      next;
    }

    push @localFiles, bsd_glob( path($filesDir)->child($trackName)
      ->child($localFile)->absolute->stringify );
  }

  return \@localFiles;
}

__PACKAGE__->meta->make_immutable;
1;
