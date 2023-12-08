# This test demonstrates that we can join a track that is not built
# Here, we join the "clinvar" track's data on refSeq, but don't build the individual clinvar track

use 5.10.0;
use strict;
use warnings;

package MockBuilder;
use Mouse;
extends 'Seq::Base';

use Test::More;
use lib 't/lib';
use TestUtils qw/ PrepareConfigWithTempdirs /;

use Path::Tiny   qw/path/;
use Scalar::Util qw/looks_like_number/;
use YAML::XS     qw/DumpFile/;

use Seq::Build;
use Seq::Tracks::Gene::Site::SiteTypeMap;
use Seq::Tracks::Reference::MapBases;

my $dir = Path::Tiny->tempdir();

my $config_file = PrepareConfigWithTempdirs(
  't/tracks/gene/join_no_build.yml',
  't/tracks/gene/db/raw', [ 'database_dir', 'files_dir', 'temp_dir' ],
  'files_dir',            $dir->stringify
);

my $config = YAML::XS::LoadFile($config_file);

my $dbPath = path( $config->{database_dir} );
$dbPath->remove_tree( { keep_root => 1 } );

my $builder = Seq::Build->new_with_config( { config => $config_file } );

my $tracks = $builder->tracksObj;

my $clinvarBuilder = $tracks->getTrackBuilderByName('clinvar');

for my $chr ( @{ $config->{chromosomes} } ) {
  ok( !$clinvarBuilder->completionMeta->_isCompleted('chrM'),
    'Clinvar track is not built' );
}

my $baseMapper = Seq::Tracks::Reference::MapBases->new();
my $siteTypes  = Seq::Tracks::Gene::Site::SiteTypeMap->new();

my $refGetter     = $tracks->getRefTrackGetter();
my $geneGetter    = $tracks->getTrackGetterByName('refSeq');
my $clinvarGetter = $tracks->getTrackGetterByName('clinvar');

my $siteTypeDbName = $geneGetter->getFieldDbName('siteType');
my $funcDbName     = $geneGetter->getFieldDbName('exonicAlleleFunction');

my $db = Seq::DBManager->new();

my $mainDbAref     = $db->dbReadAll('chrM');
my $regionDataAref = $db->dbReadAll('refSeq/chrM');

my $geneDbName = $geneGetter->dbName;
my $header     = Seq::Headers->new();

my $features = $header->getParentFeatures('refSeq');

my ( $siteTypeIdx, $funcIdx, $alleleIdIdx );

my $expectedNumberOfTracks = @{ $config->{tracks}->{tracks} } - 1;

# Check that join track successfully built
for ( my $i = 0; $i < @$features; $i++ ) {
  my $feat = $features->[$i];

  if ( $feat eq 'siteType' ) {
    $siteTypeIdx = $i;
    next;
  }

  if ( $feat eq 'exonicAlleleFunction' ) {
    $funcIdx = $i;
    next;
  }

  if ( $feat eq 'clinvar.alleleID' ) {
    $alleleIdIdx = $i;
  }
}

my $hasGeneCount = 0;
my $inGeneCount  = 0;

# We still reserve an index for all specified tracks, even if they are not built
# Therefore, to minimize database space, such tracks should be specified last
ok( !defined $clinvarGetter, 'clinvar track getter does not exist' );

for my $pos ( 0 .. $#$mainDbAref ) {
  my $dbData = $mainDbAref->[$pos];

  ok( defined $dbData, 'We have data for position ' . $pos );
  ok( @$dbData <= $expectedNumberOfTracks,
    'We don\'t have a clinvar record in database for position ' . $pos );

  my @out;
  # not an indel
  my $posIdx = 0;
  $geneGetter->get( $dbData, 'chrM', $refGetter->get($dbData), 'A', $posIdx, \@out );

  if ( $pos >= 1672 && $pos < 3230 ) {
    $inGeneCount++;

    my $alleleIDs = $out[$posIdx][$alleleIdIdx];
    my $s         = join( ",", @{ $out[$alleleIdIdx] } );

    ok( join( ",", @{ $out[$alleleIdIdx] } ) eq '24587', 'Found the clinvar record' );
    ok( join( ",", @{ $out[$siteTypeIdx] } ) eq $siteTypes->ncRNAsiteType,
      'ncRNA site type' );
    ok( !defined $out[$funcIdx][0], 'ncRNA has no exonicAlleleFunction' );
  }

  if ( defined $dbData->[$geneDbName] ) {
    $hasGeneCount++;
  }
}

ok( $inGeneCount == $hasGeneCount,
  "We have a refSeq record for every position from txStart to txEnd" );

$db->cleanUp();
$dbPath->remove_tree( { keep_root => 1 } );

done_testing();
