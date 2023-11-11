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
use DDP;

use Path::Tiny   qw/path/;
use Scalar::Util qw/looks_like_number/;
use YAML::XS     qw/LoadFile/;

# create temp directories
my $dir = Path::Tiny->tempdir();

# prepare temp directory and make test config file
my $config_file = PrepareConfigWithTempdirs(
  't/tracks/cadd/integration.yml',
  't/tracks/cadd/db', [ 'database_dir', 'files_dir', 'temp_dir' ],
  'files_dir',        $dir->stringify
);

p $config_file;

# use Seq::Tracks::Score::Build::Round;

my $runConfig = LoadFile($config_file);
p $runConfig;

my $seq = MockBuilder->new_with_config( { config => $config_file } );

my $tracks      = $seq->tracksObj;
my $caddBuilder = $tracks->getTrackBuilderByName('cadd');
my $caddGetter  = $tracks->getTrackGetterByName('cadd');
my $refBuilder  = $tracks->getTrackBuilderByName('ref');

my $db = Seq::DBManager->new();

$refBuilder->buildTrack();
$caddBuilder->buildTrack();

my @localFiles = @{ $caddBuilder->local_files };

# # adapted from scorebuilder
my $headerRegex = qr/^#/;

# # wigfix is 1-based: 1-start coordinate system in use for variableStep and fixedStep
# # https://genome.ucsc.edu/goldenpath/help/wiggle.html

my $scalingFactor = $caddGetter->scalingFactor;

my $rounder =
  Seq::Tracks::Score::Build::Round->new( { scalingFactor => $scalingFactor } );

for my $file (@localFiles) {
  my $fh = $caddBuilder->getReadFh($file);
  my $start;
  my $based = 1;

  my $pos = 0;
  my $firstRef;
  my $firstData;
  my $firstChr;

  while (<$fh>) {
    chomp;

    if ( $_ =~ m/$headerRegex/ ) {
      next;
    }

    my @fields = split "\t";

    my $chr = "chr" . $fields[0];
    my $pos = $fields[1];
    my $ref = $fields[2];
    my $alt = $fields[3];

    if ( !defined $firstRef ) {
      $firstRef = $ref;
    }

    if ( !defined $firstChr ) {
      $firstChr = $chr;
    }

    my $expected_score = $rounder->round( $fields[5] ) / $scalingFactor;

    my $data = $db->dbReadOne( $chr, $pos - 1 );

    if ( !defined $firstData ) {
      $firstData = $data;
    }

    my @out;

    $caddGetter->get( $data, $chr, $ref, $alt, 0, \@out );

    ok( @out == 1 );

    # indexed by position index (here 0, we're only checking snps atm)
    my $score = $out[0];

    say STDERR
      "chr: $chr, pos: $pos, ref: $ref, alt: $alt score: $score, expected: $expected_score";

    ok( $score == $expected_score );

    my $bystro_style_del_skipped = $caddGetter->get( $data, $chr, $ref, -10, 0, \@out );

    ok( !defined $bystro_style_del_skipped->[0] );
    ok( @{$bystro_style_del_skipped} == 1 );

    my $bystro_style_ins_skipped =
      $caddGetter->get( $data, $chr, $ref, "+ACTG", 0, \@out );

    ok( !defined $bystro_style_ins_skipped->[0] );
    ok( @{$bystro_style_ins_skipped} == 1 );

    # In some Bystro tracks, we tile across indels, outputting an annotation per base disrupted
    # For exact match tracks like CADD and VCF, we do not do this
    # And we should always only output 1 annotation per indel
    # Which is always undefined
    my $bystro_style_ins_skipped_tiling =
      $caddGetter->get( $data, $chr, $ref, "+ACTG", 0, \@out );

    ok( !defined $bystro_style_ins_skipped_tiling->[0] );
    ok( @{$bystro_style_ins_skipped_tiling} == 1 );

    $pos += 1;
  }

  # We don't currently support VCF style deletions, but we should still return undef
  my $vcf_style_del_skipped =
    $caddGetter->get( $firstData, $firstChr, $firstRef . "CTG", $firstRef, 0, [] );

  ok( !defined $vcf_style_del_skipped->[0] );
  ok( @{$vcf_style_del_skipped} == 1 );

  # We don't currently support VCF style insertions, but we should still return undef
  my $vcf_style_ins_skipped =
    $caddGetter->get( $firstData, $firstChr, $firstRef, $firstRef . "CTG", 0, [] );

  ok( !defined $vcf_style_ins_skipped->[0] );
  ok( @{$vcf_style_ins_skipped} == 1 );
}

done_testing();
