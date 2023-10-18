package TestUtils;

use 5.10.0;
use strict;
use warnings;

use Exporter 'import';
use Path::Tiny            qw(path);
use Type::Params          qw(compile);
use Types::Common::String qw(NonEmptySimpleStr);
use Types::Standard       qw(HashRef);
use YAML::XS              qw(LoadFile);

our @EXPORT_OK = qw( CopyAll UpdateConfigAttrs );

sub CopyAll {
  state $check = compile( NonEmptySimpleStr, NonEmptySimpleStr );
  my ( $src, $dest ) = $check->(@_);

  $src  = path($src);
  $dest = path($dest);

  if ( !$src->is_dir ) {
    die "Source directory does not exist";
  }

  if ( !$dest->is_dir ) {
    die "Destination directory does not exist";
  }

  # Recursive copy of directories and files from the source directory to the temporary directory
  $src->visit(
    sub {
      my ( $path, $state ) = @_;
      say $path;

      # Construct the destination path in the temporary directory
      my $this_dest = $dest->child( $path->relative($src) );

      if ( $path->is_dir ) {
        say "making $path";
        # Create directory if the current path is a directory
        $this_dest->mkpath;
      }
      else {
        # Copy the file otherwise
        $path->copy($this_dest);
        say "cp $path -> $this_dest";
      }
    },
    { recurse => 1 } # Enable recursive visiting
  );
}

sub UpdateConfigAttrs {
  state $check = compile( NonEmptySimpleStr, HashRef );
  my ( $file, $href ) = $check->(@_);

  # load config yaml
  my $config = LoadFile($file);

  # update directory keys with new location
  for my $key ( keys %{$href} ) {
    $config->{$key} = $href->{$key};
  }

  return $config;
}

1;
