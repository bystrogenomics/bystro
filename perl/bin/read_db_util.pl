#!/usr/bin/env perl

use 5.10.0;
use strict;
use warnings;

package MockBuilder;
use Mouse;
extends 'Seq::Base';

1;

#Supply config, track, chromosome, start, stop
use Scalar::Util qw/looks_like_number/;
use DDP;

my $config = $ARGV[0];
my $track  = $ARGV[1];
my $chrom  = $ARGV[2];
my $start  = $ARGV[3];
my $stop   = $ARGV[4];

if ( !looks_like_number($start) || !looks_like_number($stop) ) {
  die 'config<path/to/yaml> trackName<str> chrom<str> start<int> stop<int>';
}

my $seq = MockBuilder->new_with_config( { config => $config, readOnly => 1 } );

my $tracks = $seq->tracksObj;

my $refTrackGetter = $tracks->getRefTrackGetter();
my $trackGetter    = $tracks->getTrackGetterByName($track);

my $isRef;
if ( $refTrackGetter eq $trackGetter ) {
  $isRef = 1;
}

if ( !$trackGetter ) {
  die "$track not found in config $config";
}

my $db = Seq::DBManager->new();

my @positions = ( $start .. $stop );
my @results   = ( $start .. $stop );

my $data = $db->dbRead( $chrom, \@results );

say STDERR "Read data for $chrom $start";
p $data;
say STDERR "Showing results for $track";
my %alt = ( 'A' => 'T', 'C' => 'G', 'G' => 'C', 'T' => 'A' );

my $idx = 0;
for my $d (@$data) {
  my $pos = $positions[$idx];
  $idx++;

  if ($isRef) {
    say $refTrackGetter->get($d);

    next;
  }

  my $ref = $refTrackGetter->get($d);

  for my $alt (qw/A C G T/) {
    if ( $alt eq $ref ) {
      next;
    }

    my $out = [];
    say STDERR "$chrom:$pos, ref: $ref, alt: $alt:";

    $trackGetter->get( $d, $chrom, $ref, $alt, 0, $out, $start - 1 );

    p $out;
  }
}

# if($trackGetter->type eq 'gene') {
#   p $trackGetter->{_db}->dbReadAll( $trackGetter->regionTrackPath($chrom) );
# }
