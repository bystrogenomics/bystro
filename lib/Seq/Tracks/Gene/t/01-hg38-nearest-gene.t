use 5.10.0;
use warnings;
use strict;

package MockAnnotationClass;
use lib './lib';
use Mouse;
extends 'Seq::Base';
use Seq::Tracks;

has tracks => (is => 'ro', required => 1);
1;

package TestRead;
use DDP;

use Test::More;
use Seq::DBManager;
use MCE::Loop;

my $class = MockAnnotationClass->new_with_config({ config => './config/hg38.yml'});

my $tracks = Seq::Tracks->new({tracks => $class->tracks, gettersOnly => 1});

my $db = Seq::DBManager->new();

# Set the lmdb database to read only, remove locking
# We MUST make sure everything is written to the database by this point
$db->setReadOnly(1);
  
my $geneTrack = $tracks->getTrackGetterByName('refSeq');

my $dataHref;

my $geneTrackIdx = $geneTrack->dbName;
my $nearestTrackIdx = $geneTrack->nearestDbName;

MCE::Loop::init {
   max_workers => 8, chunk_size => 1
};

my @allChrs = $geneTrack->allWantedChrs();

plan tests => scalar @allChrs;

mce_loop {
  my ($mce, $chunk_ref, $chunk_id) = @_;
  my $chr = $_;
  # TODO: Remove in next iteration
  if($chr eq 'chrM') {
    say "skpping chrM";
    return;
  }

  say "examining chr $chr";

  my $lastDbPos = $db->dbGetNumberOfEntries($chr) - 1;

  my $foundErr;
  for my $dbPos (0 .. $lastDbPos) {
    $dataHref = $db->dbReadOne($chr, $dbPos);

    if(!defined $dataHref->[$geneTrackIdx] && !defined $dataHref->[$nearestTrackIdx]) {
      say STDERR "$chr: $dbPos (0-based) has no nearest gene data or gene track data";
      $foundErr = 1;
    }
  }

  ok(!defined $foundErr, "chr $chr has nearest gene data at every position, from 0 .. $lastDbPos");
} @allChrs;

MCE::Loop::finish();