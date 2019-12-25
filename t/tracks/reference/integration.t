use 5.10.0;
use strict;
use warnings;

package MockBuilder;
use lib './lib';
use Mouse;
extends 'Seq::Base';

1;

use Test::More;
use Path::Tiny qw/path/;
use Scalar::Util qw/looks_like_number/;
use YAML::XS qw/LoadFile/;

my $config = './t/tracks/reference/integration.yml';
my $runConfig = LoadFile($config);

my %wantedChr = map { $_ => 1 } @{ $runConfig->{chromosomes} };

my $seq = MockBuilder->new_with_config({config => path($config )->absolute, debug => 1});

path($seq->database_dir)->remove_tree({keep_root => 1});

my $tracks = $seq->tracksObj;

my $refBuilder = $tracks->getRefTrackBuilder();
my $refGetter = $tracks->getRefTrackGetter();

my $db = Seq::DBManager->new();

$refBuilder->buildTrack();

my @localFiles = @{$refBuilder->local_files};

for my $file (@localFiles) {
  my $fh = $refBuilder->getReadFh($file);
  my ($chr, $pos);

  while(<$fh>) {
    chomp;

    if($_ =~ m/>(\S+)/) {
      $chr = $1;

      $pos = 0;

      next;
    }

    if(!$wantedChr{$chr}) {
      next;
    }

    for my $base (split '', $_) {
      my $data = $db->dbReadOne($chr, $pos);
      my $out = [];

      my $dbBase = $refGetter->get($data);

      ok(uc($base) eq $dbBase);

      $pos++;
    }
  }
}

path($seq->database_dir)->remove_tree({keep_root => 1});

done_testing();