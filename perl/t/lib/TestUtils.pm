package TestUtils;

use 5.10.0;
use strict;
use warnings;

use Exporter 'import';
use Path::Tiny            qw(path);
use Type::Params          qw(compile);
use Types::Common::String qw(NonEmptySimpleStr);
use Types::Standard       qw(ArrayRef HashRef);
use YAML::XS              qw(DumpFile LoadFile);

our @EXPORT_OK =
  qw( CopyAll HaveRequiredBinary PrepareConfigWithTempdirs UpdateConfigAttrs );

sub HaveRequiredBinary {
  my $binary         = shift;
  my $path_to_binary = `which $binary`;
  chomp($path_to_binary); # Remove trailing newline, if any
  if ($path_to_binary) {
    return 1;
  }
  else {
    return;
  }
}

# PrepareConfigWithTempdirs takes parameters below and returns a string to a
#   temporary config file with updated paths in that config file and returns
#   an absolute path to the config file
# config_file => configuration file
# src_dir => directory of raw files needed for the test
# want_dest_dirs => names of directories that will be created
# target_dir => name of directory for the raw data
# dest_dir => destination directory
sub PrepareConfigWithTempdirs {
  state $check = compile(
    NonEmptySimpleStr,            NonEmptySimpleStr,
    ArrayRef [NonEmptySimpleStr], NonEmptySimpleStr,
    NonEmptySimpleStr
  );
  my ( $config_file, $src_dir, $want_dest_dirs, $target_dir, $dest_dir ) =
    $check->(@_);

  my %tempDirsForWantDir;

  for my $dir (@$want_dest_dirs) {
    my $d = path($dest_dir)->child($dir);
    $d->mkpath;
    $tempDirsForWantDir{$dir} = $d->stringify;
  }

  # copy files into temporary dir
  CopyAll( $src_dir, $tempDirsForWantDir{$target_dir} );

  # update config to include temp directories
  my $test_config = UpdateConfigAttrs( $config_file, \%tempDirsForWantDir );

  # write new test config to file
  my $test_config_file = path($dest_dir)->child('config.yml');
  DumpFile( $test_config_file, $test_config );

  return $test_config_file->absolute->stringify;
}

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

      # Construct the destination path in the temporary directory
      my $this_dest = $dest->child( $path->relative($src) );

      if ( $path->is_dir ) {
        # Create directory if the current path is a directory
        $this_dest->mkpath;
      }
      else {
        # Copy the file otherwise
        $path->copy($this_dest);
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
