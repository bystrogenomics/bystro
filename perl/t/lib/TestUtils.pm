package TestUtils;

use 5.10.0;
use strict;
use warnings;

use Exporter 'import';
use Path::Tiny;
use YAML::XS qw/LoadFile/;

use Type::Params          qw(compile);
use Types::Standard       qw(HashRef);
use Types::Common::String qw(NonEmptySimpleStr);

our @EXPORT_OK = qw/ UpdateConfigAttrs /;

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
