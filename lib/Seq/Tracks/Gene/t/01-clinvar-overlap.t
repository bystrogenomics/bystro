use 5.10.0;
use strict;
use warnings;
use lib './lib';


package MockBuild;
use Mouse 2;
use Seq::Tracks::Build;
extends "Seq::Base";
#TODO: allow building just one track, identified by name
has config => (is => 'ro', isa => 'Str', required => 1);

package MockAnnotate;
use lib './lib';
use Test::More;
use DDP;
use Seq::DBManager;
use Seq;

my $mock = MockBuild->new_with_config({config => './lib/Seq/Tracks/Gene/t/clinvar-test-config.yml', chromosomes => ['chrY'], verbose => 0 });

my $trackBuilder = $mock->tracksObj->getTrackBuilderByName('refSeq');

p $trackBuilder;

$trackBuilder->buildTrack();

my $refSeqTrackGetter = $mock->tracksObj->getTrackGetterByName('refSeq');

# The config file has these features for refSeq.clinvar
  # - alleleID: number
  # - phenotypeList
  # - clinicalSignificance
  # - reviewStatus
  # - type
  # - chromStart
  # - chromEnd

my $db = Seq::DBManager->new();

my $dataAref = $db->dbReadOne('chrY', 14523504 - 1);

my $refSeqTrackIdx = $refSeqTrackGetter->dbName;

# p $clinvarTrackIndex;

my $refSeqDataAref = $dataAref->[$refSeqTrackIdx];
my @refSeqData = @$refSeqDataAref;
p @refSeqData;

done_testing();
1;