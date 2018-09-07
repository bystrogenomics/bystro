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
use DDP;
use Seq::Tracks::Gene::Site::SiteTypeMap;
use Seq::Tracks::Reference::MapBases;
use Seq::DBManager;

my $baseMapper = Seq::Tracks::Reference::MapBases->new();
my $siteTypeMap = Seq::Tracks::Gene::Site::SiteTypeMap->new();

# Defines three tracks, a nearest gene , a nearest tss, and a region track
# The region track is simply a nearest track for which we storeOverlap and do not storeNearest
# To show what happens when multiple transcripts (as in NR_FAKE3, NR_FAKE3B, NR_FAKE3C)
# all share 100% of their data, except have different txEnd's, which could reveal issues with our uniqueness algorithm
# such as calculating the maximum range of the overlap: in previous code iterations
# we removed the non-unique overlapping data, without first looking at the txEnd
# and therefore had a smaller-than-expected maximum range
my $seq = MockBuilder->new_with_config({config => './t/tracks/gene/overlap.yml', debug => 1});
my $tracks = $seq->tracksObj;

my $dbPath = path($seq->database_dir);
$dbPath->remove_tree({keep_root => 1});

###### First make fake reference track, using the column of sequence data #####

my $refBuilder = $tracks->getRefTrackBuilder();
my $refIdx = $refBuilder->dbName;

$refBuilder->buildTrack();

my $db = Seq::DBManager->new();

my $geneBuilder = $tracks->getTrackBuilderByName('refSeq');

my $refGetter = $tracks->getRefTrackGetter();
my $geneGetter = $tracks->getTrackGetterByName('refSeq');

my $siteTypeDbName = $geneGetter->getFieldDbName('siteType');
my $funcDbName = $geneGetter->getFieldDbName('exonicAlleleFunction');

$geneBuilder->buildTrack();

### We have:

my $regionDataAref = $db->dbReadAll('refSeq/chr10');

my $geneIdx = $geneBuilder->dbName;
my $header = Seq::Headers->new();

my $features = $header->getParentFeatures('refSeq');

my ($siteTypeIdx, $funcIdx, $nameIdx, $name2Idx, $spDispIdx, $mrnaIdx, $spIdx, $descIdx);

my $dbLen = $db->dbGetNumberOfEntries('chr10');


for (my $i = 0; $i < @$features; $i++) {
  my $feat = $features->[$i];

  if($feat eq 'siteType') {
    $siteTypeIdx = $i;
    next;
  }

  if($feat eq 'exonicAlleleFunction') {
    $funcIdx = $i;
    next;
  }

  if($feat eq 'name') {
    $nameIdx = $i;
    next;
  }

  if($feat eq 'name2') {
    $name2Idx = $i;
    next;
  }

  if($feat eq 'mRNA') {
    $mrnaIdx = $i;
    next;
  }

  if($feat eq 'spID') {
    $spIdx = $i;
    next;
  }

  if($feat eq 'spDisplayID') {
    $spDispIdx = $i;
    next;
  }

  if($feat eq 'description') {
    $descIdx = $i;
    next;
  }
}

# Safe for use when instantiated to static variable; no set - able properties
my $coding = $siteTypeMap->codingSiteType;
my $utr5 = $siteTypeMap->fivePrimeSiteType;
my $utr3 = $siteTypeMap->threePrimeSiteType;
my $spliceAcceptor = $siteTypeMap->spliceAcSiteType;
my $spliceDonor = $siteTypeMap->spliceDonSiteType;
my $ncRNA = $siteTypeMap->ncRNAsiteType;
my $intronic = $siteTypeMap->intronicSiteType;

#                   txStart  txEnd   cdsStart   cdsEnd    exonStarts          exonEnds
# NR_033266 chr19 - 60950    70966   70966      70966  3  60950,66345,70927,  61894,66499,70966,

# NM_001009943 index; has some overlapping values in mRNA, rfamAcc, description
my $targetIdx;
# NM_001009941
my $target2Idx;
my $expSpIds = join("\t", sort { $a cmp $b } ('F8WEI4', 'Q6P6B7'));

my $expDesc = join("\t", sort { $a cmp $b } (
  'Homo sapiens ankyrin repeat domain 16 (ANKRD16), transcript variant 4, mRNA.',
  'Homo sapiens ankyrin repeat domain 16 (ANKRD16), transcript variant 1, mRNA.'
));

my $exp2SpIds = 'Q6P6B7';

my $exp2Desc = 'Homo sapiens ankyrin repeat domain 16 (ANKRD16), transcript variant 2, mRNA.';

# We're always only getting a "snp", so posIdx is 0 for the allele
my $posIdx = 0;
for my $pos (0 .. $dbLen - 1) {
  my $mainDbAref = $db->dbReadOne('chr10', $pos);

  my $refBase = $refGetter->get($mainDbAref);
  my $alt = 'A';

  my $out = [];
  my $refSeqData = $geneGetter->get($mainDbAref, 'chr10', $refBase, $alt, $posIdx, $out);

  my $siteType = $out->[$siteTypeIdx][$posIdx];
  my $names = $out->[$nameIdx][$posIdx];
  my $symbol = $out->[$name2Idx][$posIdx];

  # Exon is NM_019046, none others overlap
  if($pos < 16361) {
    ok(!ref $siteType);

    # last exon is utr3; this tx is on negative strand so first exon is really last
    # exonStarts: 0,16371,18572,21280,22243,23997,26121,27315,
    # exonEnds: 966,16562,18651,21442,22352,24040,26342,28172
    # also, 966 is exonEnds, which are open interval, so last base is -1
    if($pos <= 966 - 1) {
      ok($siteType eq $utr3);
      next;
    }

    # closing boundary not needed since > 16361, for clarity only
    # 16371 is exonStarts[1] which is closed interval (so no - 1 to get first base)
    if($pos > 966 - 1 && $pos < 16371) {
      if($pos == 966 || $pos == 967) {
        # First 2 bases are really the last; so first 2 should be spliceAcceptor
        # when on negative strand, instead of spliceDonor
        ok($siteType eq $spliceAcceptor);
        next;
      }

      ok($siteType eq $intronic);
    }

     next;
  }

  # first condition not needed, simply for consistency
  # 1652 is the first exonEnds for NM_001009943 and NM_001009941, and the 2nd for NM_019046
  if($pos >= 16361 && $pos <= 16562 - 1) {
    if(!$targetIdx) {
      my $idx = 0;

      for my $name (@$names) {
        if($name eq 'NM_001009943') {
          $targetIdx = $idx;
        } elsif($name eq 'NM_001009941') {
          $target2Idx = $idx;
        }

        $idx++
      }
    }

    my $spIds = $out->[$spIdx][$posIdx][$targetIdx];
    my $descs = $out->[$descIdx][$posIdx][$targetIdx];
    ok(join("\t", sort { $a cmp $b } @$spIds) eq $expSpIds);
    ok(join("\t", sort { $a cmp $b } @$descs) eq $expDesc);

    my $spIds2 = $out->[$spIdx][$posIdx][$target2Idx];
    my $descs2 = $out->[$descIdx][$posIdx][$target2Idx];

    ok($spIds2 eq $exp2SpIds);
    ok($descs2 eq $exp2Desc);
  }
#   ok($out->[$siteTypeIdx][0] eq $intronic);
}

# ok($inGeneCount == $hasGeneCount, "We have a refSeq record for every position from txStart to txEnd");

$db->cleanUp();
$dbPath->remove_tree({keep_root => 1});

done_testing();
