use 5.10.0;
use strict;
use warnings;

package MockBuilder;
use Mouse;
extends 'Seq::Base';

1;

use Test::More;
use Test::Exception;
use lib 't/lib';
use TestUtils qw/ PrepareConfigWithTempdirs /;

use Path::Tiny   qw/path/;
use Scalar::Util qw/looks_like_number/;
use YAML::XS     qw/DumpFile/;

use Seq::Build;
# create temp directories
my $dir = Path::Tiny->tempdir();

# prepare temp directory and make test config file
my $config_file = PrepareConfigWithTempdirs(
  't/tracks/build/ref_cannot_be_skipped.yml',
  't/tracks/gene/db/raw', [ 'database_dir', 'files_dir', 'temp_dir' ],
  'files_dir',            $dir->stringify
);

my $config = YAML::XS::LoadFile($config_file);

throws_ok { Seq::Build->new_with_config( { config => $config_file } ) } qr/Reference track is marked as no_build, but must be built/, 'Reference tracks must be built';

done_testing();