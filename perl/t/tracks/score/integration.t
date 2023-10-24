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

use Path::Tiny   qw/path/;
use Scalar::Util qw/looks_like_number/;
use YAML::XS     qw/LoadFile/;

# create temp directories
my $dir = Path::Tiny->tempdir();

# prepare temp directory and make test config file
my $config_file = PrepareConfigWithTempdirs(
  't/tracks/score/integration.yml',
  't/tracks/score/db/raw', [ 'database_dir', 'files_dir', 'temp_dir' ],
  'files_dir',             $dir->stringify
);

use Seq::Tracks::Score::Build::Round;

my $runConfig = LoadFile($config_file);

my %wantedChr = map { $_ => 1 } @{ $runConfig->{chromosomes} };

my $seq = MockBuilder->new_with_config( { config => $config_file } );

my $tracks       = $seq->tracksObj;
my $scoreBuilder = $tracks->getTrackBuilderByName('phastCons');
my $scoreGetter  = $tracks->getTrackGetterByName('phastCons');

my $db = Seq::DBManager->new();

$scoreBuilder->buildTrack();

my @localFiles = @{ $scoreBuilder->local_files };

# adapted from scorebuilder
my $headerRegex = qr/^(fixedStep)\s+chrom=(\S+)\s+start=(\d+)\s+step=(\d+)/;

# wigfix is 1-based: 1-start coordinate system in use for variableStep and fixedStep
# https://genome.ucsc.edu/goldenpath/help/wiggle.html

my $scalingFactor = $scoreGetter->scalingFactor;

my $rounder =
  Seq::Tracks::Score::Build::Round->new( { scalingFactor => $scalingFactor } );

for my $file (@localFiles) {
  my $fh = $scoreBuilder->getReadFh($file);
  my $step;
  my $pos;
  my $chr;
  my $start;
  my $based = 1;

  while (<$fh>) {
    chomp;

    if ( $_ =~ m/$headerRegex/ ) {
      $chr = $2;

      $step = $4;

      $start = $3;

      $pos = $start - $based;

      next;
    }

    if ( !$wantedChr{$chr} ) {
      next;
    }

    my $value = $_;

    my $rounded = $rounder->round($_) / $scalingFactor;

    my $data = $db->dbReadOne( $chr, $pos );

    my $out = [];

    $scoreGetter->get( $data, $chr, 'C', 'T', 0, $out );

    # indexed by position index (here 0, we're only checking snps atm)
    my $score = $out->[0];

    ok( $score == $rounded );

    # comes after, because first position after header is $start
    $pos += $step;
  }
}

done_testing();
