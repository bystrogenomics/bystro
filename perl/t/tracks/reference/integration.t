use 5.10.0;
use strict;
use warnings;

package MockBuilder;

use Mouse;
extends 'Seq::Base';

1;

use Test::More;
use lib 't/lib';
use TestUtils qw/ PrepareConfigWithTempdirs /;

use Path::Tiny;
use Scalar::Util qw/looks_like_number/;
use YAML::XS     qw/ LoadFile /;

# create temp directories
my $dir = Path::Tiny->tempdir();

# prepare temp directory and make test config file
my $config_file = PrepareConfigWithTempdirs(
  't/tracks/reference/integration.yml',
  't/tracks/reference/db/raw', [ 'database_dir', 'files_dir', 'temp_dir' ],
  'files_dir',                 $dir->stringify
);

# get chromosomes that are considered
my $runConfig = LoadFile($config_file);
my %wantedChr = map { $_ => 1 } @{ $runConfig->{chromosomes} };

my $seq = MockBuilder->new_with_config( { config => $config_file } );

my $tracks     = $seq->tracksObj;
my $refBuilder = $tracks->getRefTrackBuilder();
my $refGetter  = $tracks->getRefTrackGetter();
my $db         = Seq::DBManager->new();

$refBuilder->buildTrack();

my @localFiles = @{ $refBuilder->local_files };

for my $file (@localFiles) {
  my $fh = $refBuilder->getReadFh($file);
  my ( $chr, $pos );

  while (<$fh>) {
    chomp;

    if ( $_ =~ m/>(\S+)/ ) {
      $chr = $1;
      $pos = 0;
      next;
    }

    if ( !$wantedChr{$chr} ) {
      next;
    }

    for my $base ( split '', $_ ) {
      my $data   = $db->dbReadOne( $chr, $pos );
      my $out    = [];
      my $dbBase = $refGetter->get($data);

      ok( uc($base) eq $dbBase );

      $pos++;
    }
  }
}

done_testing();
