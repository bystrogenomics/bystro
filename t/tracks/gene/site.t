use 5.10.0;
use strict;
use warnings;
package Testing;

use Test::More;
use DDP;
use lib './lib';

# plan tests => 13;

use Seq::Tracks::Gene::Site;

my $siteHandler = Seq::Tracks::Gene::Site->new();

my $packedData = $siteHandler->pack(
  (0, 'intronic', '-')
);

say "Packed data for '(0, intronic, -)' is ";

my ($txNumber, $unpackedData) = $siteHandler->unpack($packedData);

my ($strand, $siteType, $codonNumber, $codonPosition, $codonSequence);

($strand, $siteType) = @$unpackedData;

p $unpackedData;
ok($txNumber == 0, 'returns txNumber');
ok($strand eq '-', 'reads strand ok');
ok($siteType eq 'intronic', 'reads intronic ok from shortened site');
ok(@$unpackedData == 2, 'intronic sites have only siteType and strand');


$packedData = $siteHandler->pack(
  (1541, 'ncRNA', '+')
);

($txNumber, $unpackedData) = $siteHandler->unpack($packedData);

($strand, $siteType) = @$unpackedData;

ok($txNumber == 1541, 'returns txNumber');
ok($strand eq '+', 'reads strand ok');
ok($siteType eq 'ncRNA', 'reads intronic ok from shortened site');
ok(@$unpackedData == 2, 'intronic sites have only siteType and strand');

p $unpackedData;

$packedData = $siteHandler->pack(
  (65000,'exonic', '+', 1, 2, 'ATG')
);

say "Packed data for (65000, 'Coding', '+', 1, 2, 'ATG') is ";
p $packedData;

($txNumber, $unpackedData) = $siteHandler->unpack($packedData);

($strand, $siteType, $codonNumber, $codonPosition, $codonSequence) = @$unpackedData;

ok($txNumber == 65000, 'returns txNumber');
ok($strand eq '+', 'reads strand ok');
ok($siteType eq 'exonic', 'reads intronic ok from shortened site');
ok($codonNumber == 1, 'reads codonPosition');
ok($codonPosition == 2, 'reads codonNumber');
ok($codonSequence eq 'ATG', 'reads codon sequnce');
ok(@$unpackedData == 5, 'intronic sites have only siteType and strand');

p $unpackedData;

done_testing();