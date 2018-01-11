# Class; requires config file
# Regions we need to prepare
# 25115000        25170687  
# 19863040        19929515 
# 19881779        19929515    
# 18852555        18881944 
# 19863040        19929515 

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
use DDP;

# my $baseMapper = Seq::Tracks::Reference::MapBases->new();

# Defines three tracks, a nearest gene , a nearest tss, and a region track
# The region track is simply a nearest track for which we storeOverlap and do not storeNearest
# To show what happens when multiple transcripts (as in NR_FAKE3, NR_FAKE3B, NR_FAKE3C) 
# all share 100% of their data, except have different txEnd's, which could reveal issues with our uniqueness algorithm
# such as calculating the maximum range of the overlap: in previous code iterations
# we removed the non-unique overlapping data, without first looking at the txEnd
# and therefore had a smaller-than-expected maximum range
my $seq = MockBuilder->new_with_config({config => './t/tracks/gene/test.yml', debug => 1});

# system('rm -rf ' . path($seq->database_dir)->child('*'));

my $tracks = $seq->tracksObj;
my $refBuilder = $tracks->getRefTrackBuilder();
my $geneBuilder = $tracks->getTrackBuilderByName('refSeq');

# $refBuilder->buildTrack();

my $refGetter = $tracks->getRefTrackGetter();
my $db = Seq::DBManager->new({delete => 1});

$db->dbPatch('chr21', $geneBuilder->dbName, 9825831);
my $stuff = $db->dbReadOne('chr21', 9825831);
p $stuff;
# exit;
my $dbLength = $db->dbGetNumberOfEntries('chr21');

my $cursor = $db->dbStartCursorTxn('chr21');

for my $i (0 .. $dbLength) {
  $db->dbPatchSequential($cursor, 'chr21', $geneBuilder->dbName, $i);
}

$db->cleanUp();

$geneBuilder->buildTrack();
