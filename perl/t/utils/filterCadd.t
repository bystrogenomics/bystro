use strict;
use warnings;
use 5.10.0;

use Test::More;
use lib 't/lib';
use TestUtils qw/ CopyAll /;

use Path::Tiny;
use YAML::XS qw/LoadFile DumpFile/;

use Seq::DBManager;
use Seq::Tracks::Cadd;
use Seq::Tracks::Reference;
use Seq::Tracks::Score::Build::Round;
use Utils::FilterCadd;

# create temp directories
my $db_dir   = Path::Tiny->tempdir();
my $base_dir = Path::Tiny->tempdir();
my $raw_dir  = $base_dir->child('raw');

$raw_dir->mkpath;

# Copy test contents to temporary directory
CopyAll( 't/utils/raw', $raw_dir->stringify );

my $config = {
  'assembly'     => 'hg38',
  'chromosomes'  => [ 'chr22', 'chr1', 'chr2' ],
  'database_dir' => $db_dir->stringify,
  'files_dir'    => $raw_dir->stringify,
  'tracks'       => {
    'tracks' => [
      {
        'assembly'    => 'hg38',
        'chromosomes' => [ 'chr22', 'chr1', 'chr2' ],
        'name'        => 'ref',
        'type'        => 'reference'
      },
      {
        'assembly'             => 'hg38',
        'build_date'           => '2017-04-22T05:22:00',
        'caddToBed_date'       => '2017-01-19T04:37:00',
        'chromosomes'          => [ 'chr22', 'chr1', 'chr2' ],
        'fetch_completed'      => '2023-10-17T21:55:00',
        'filterCadd_completed' => '2023-05-26T14:05:00',
        'filterCadd_date'      => '2017-09-12T19:18:00',
        'local_files'          => [
          'test.filterCadd.cadd.chr22.txt', 'test.filterCadd.cadd.chr1.txt.gz',
          'test.filterCadd.cadd.chr2.txt'
        ],
        'name'          => 'cadd',
        'sortCadd_date' => '2017-01-20T23:53:00',
        'sorted'        => 1,
        'type'          => 'cadd'
      }
    ]
  }
};

# write temporary config file
my $config_file = $base_dir->child('filterCadd.yml');
DumpFile( $config_file, $config );

Seq::DBManager::initialize( { databaseDir => $config->{database_dir}, } );

my $db = Seq::DBManager->new();

my $ref  = Seq::Tracks::Reference->new( $config->{tracks}{tracks}[0] );
my $cadd = Seq::Tracks::Cadd->new( $config->{tracks}{tracks}[1] );

my $rounder = Seq::Tracks::Score::Build::Round->new( { scalingFactor => 10 } );

# chr22   10584987        10584988        C       A       -0.003351       2.554
# chr22   10584987        10584988        C       G       -0.145476       1.416
# chr22   10584987        10584988        C       T       -0.050851       2.124
$db->dbPatch( 'chr22', $cadd->dbName, 10584987,
  [ $rounder->round(2.554), $rounder->round(1.416), $rounder->round(2.124) ] );

# chr1    10005   10006   C       A       0.185685        4.528
# chr1    10005   10006   C       G       -0.025782       2.345
# chr1    10005   10006   C       T       0.089343        3.494
$db->dbPatch( 'chr1', $cadd->dbName, 10005,
  [ $rounder->round(4.528), $rounder->round(2.345), $rounder->round(3.494) ] );

# chr2    10002   10003   T       A       0.370069        6.349
# chr2    10002   10003   T       C       0.094635        3.551
# chr2    10002   10003   T       G       0.210401        4.788
$db->dbPatch( 'chr2', $cadd->dbName, 10002,
  [ $rounder->round(6.349), $rounder->round(3.551), $rounder->round(4.788) ] );

# $db->dbPatch('chr22', $ref->dbName, 10584987, 2);
# $db->dbPatch('chr22', $ref->dbName, 10584988, 3);
# $db->dbPatch('chr22', $ref->dbName, 10584989, 1);
# $db->dbPatch('chr22', $ref->dbName, 10584990, 3);
# $db->dbPatch('chr22', $ref->dbName, 10584991, 4);
# $db->dbPatch('chr22', $ref->dbName, 10584992, 4);
# $db->dbPatch('chr22', $ref->dbName, 10584993, 3);

my $filter = Utils::FilterCadd->new(
  {
    config     => $config_file,
    name       => 'cadd',
    maxThreads => 1,
    utilName   => 'fetch',
    compress   => 0
  }
);

my $success = $filter->go();

my $fh = $raw_dir->child('cadd/test.filterCadd.cadd.chr22.chr22.filtered.txt')
  ->filehandle('<');

my $header = <$fh>;
$header .= <$fh>;

my $count = 0;
while (<$fh>) {
  chomp;

  my @fields = split '\t', $_;

  ok( $fields[0] eq 'chr22',   "maintains chrom" );
  ok( $fields[1] == 10584987,  "maintains chromStart" );
  ok( $fields[-4] eq "C",      "maintains ref" );
  ok( $fields[-5] == 10584988, "maintains chromEnd" );

  if ( $. == 3 ) {
    ok( $fields[-1] == 2.554, "keeps 1st allele score order" );
    ok( $fields[-3] eq "A",   "keeps 1st allele order" );
  }
  elsif ( $. == 4 ) {
    ok( $fields[-1] == 1.416, "keeps 2nd allele score order" );
    ok( $fields[-3] eq "G",   "keeps 2nd allele order" );
  }
  elsif ( $. == 5 ) {
    ok( $fields[-1] == 2.124, "keeps 3rd allele score order" );
    ok( $fields[-3] eq "T",   "keeps 3rd allele order" );
  }

  $count = $.;
}

close($fh);

ok( $count == 5, "found expectd number of lines" );

ok( $success == 1, "exited cleanly" );

$config = LoadFile( $config_file->stringify );

my $caddTrack = $config->{tracks}{tracks}[1];

$fh = $raw_dir->child('cadd/test.filterCadd.cadd.chr1.chr1.filtered.txt')
  ->filehandle('<');

$header = <$fh>;
$header .= <$fh>;

$count = 0;
while (<$fh>) {
  chomp;

  my @fields = split '\t', $_;

  ok( $fields[0] eq 'chr1', "maintains chrom" );
  ok( $fields[1] == 10005,  "maintains chromStart" );
  ok( $fields[-4] eq "C",   "maintains ref" );
  ok( $fields[-5] == 10006, "maintains chromEnd" );

  if ( $. == 3 ) {
    ok( $fields[-1] == 4.528, "keeps 1st allele score order" );
    ok( $fields[-3] eq "A",   "keeps 1st allele order" );
  }
  elsif ( $. == 4 ) {
    ok( $fields[-1] == 2.345, "keeps 2nd allele score order" );
    ok( $fields[-3] eq "G",   "keeps 2nd allele order" );
  }
  elsif ( $. == 5 ) {
    ok( $fields[-1] == 3.494, "keeps 3rd allele score order" );
    ok( $fields[-3] eq "T",   "keeps 3rd allele order" );
  }

  $count = $.;
}

close($fh);

ok( $count == 5, "found expected number of lines" );

ok( $success == 1, "exited cleanly" );

$config = LoadFile( $config_file->stringify );

$caddTrack = $config->{tracks}{tracks}[1];

$fh = $raw_dir->child('cadd/test.filterCadd.cadd.chr2.chr2.filtered.txt')
  ->filehandle('<');

$header = <$fh>;
$header .= <$fh>;

$count = 0;
while (<$fh>) {
  chomp;

  my @fields = split '\t', $_;

  ok( $fields[0] eq 'chr2', "maintains chrom" );
  ok( $fields[1] == 10002,  "maintains chromStart" );
  ok( $fields[-4] eq "T",   "maintains ref" );
  ok( $fields[-5] == 10003, "maintains chromEnd" );

  if ( $. == 3 ) {
    ok( $fields[-1] == 6.349, "keeps 1st allele score order" );
    ok( $fields[-3] eq "A",   "keeps 1st allele order" );
  }
  elsif ( $. == 4 ) {
    ok( $fields[-1] == 3.551, "keeps 2nd allele score order" );
    ok( $fields[-3] eq "C",   "keeps 2nd allele order" );
  }
  elsif ( $. == 5 ) {
    ok( $fields[-1] == 4.788, "keeps 3rd allele score order" );
    ok( $fields[-3] eq "G",   "keeps 3rd allele order" );
  }

  $count = $.;
}

close($fh);

ok( $count == 5, "found expected number of lines" );

ok( $success == 1, "exited cleanly" );

$config = LoadFile( $config_file->stringify );

$caddTrack = $config->{tracks}{tracks}[1];

ok( $caddTrack->{filterCadd_date}, "has non-null filterCadd_date property" );

is_deeply(
  $caddTrack->{local_files},
  [
    "test.filterCadd.cadd.chr22.chr22.filtered.txt",
    "test.filterCadd.cadd.chr1.chr1.filtered.txt",
    "test.filterCadd.cadd.chr2.chr2.filtered.txt"
  ],
  "expected filtered CADD files."
);

done_testing();
