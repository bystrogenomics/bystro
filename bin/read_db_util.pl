use 5.10.0;
use strict;
use warnings;

package MockBuilder;
use lib './lib';
use Mouse;
extends 'Seq::Base';

1;

#Supply config, track, chromosome, start, stop
use Scalar::Util qw/looks_like_number/;
use DDP;

my $config = $ARGV[0];
my $track = $ARGV[1];
my $chrom = $ARGV[2];
my $start = $ARGV[3];
my $stop = $ARGV[4];

if(!looks_like_number($start) && looks_like_number($stop)) {
  die '3rd and 4th args should be numbers (start,stop) 0-based';
}

my $seq = MockBuilder->new_with_config({config => $config, readOnly => 1});

my $tracks = $seq->tracksObj;

my $refTrackGetter = $tracks->getRefTrackGetter();
my $trackGetter = $tracks->getTrackGetterByName($track);

my $isRef;
if($refTrackGetter eq $trackGetter) {
  $isRef = 1;
}

if(!$trackGetter) {
  die "$track not found in config $config";
}

my $db = Seq::DBManager->new();

my $data = $db->dbRead($chrom, [$start .. $stop]);

for my $d (@$data) {
  if($isRef) {
    say $refTrackGetter->get($d);

    next;
  }

  p $d->[$trackGetter->dbName];
}