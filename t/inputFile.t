use 5.10.0;
use strict;
use warnings;
use lib './lib';

use Seq::InputFile;
use Test::More;
use DDP;

my $inputter = Seq::InputFile->new();

my $err = $inputter->checkInputFileHeader(["Chrom", "Pos", "Ref", "Alt", "Type"]);

ok(!defined $err);

$err = $inputter->checkInputFileHeader(["Fragment", "Position", "Reference", "Alleles", "Type"]);

ok(!defined $err);

$err = $inputter->checkInputFileHeader(["Fragment", "Position", "Reference", "Minor_alleles", "Type"]);

ok(!defined $err);

$err = $inputter->checkInputFileHeader(["Fragment", "Position", "Reference", "Type", "Alt"]);

ok(!defined $err, "Alt, Type order doesn't matter");

$err = $inputter->checkInputFileHeader(["Position", "Fragment", "Reference", "Type", "Alt"]);

ok($err, "Chrom, Pos, Ref order matters");

$err = $inputter->checkInputFileHeader(["Type", "Alt"]);

ok($err, "Chrom, Pos, Ref required");

$err = $inputter->checkInputFileHeader(["Ref", "Type", "Alt"]);

ok($err, "Chrom, Pos, Ref required");

$err = $inputter->checkInputFileHeader(["Pos,", "Ref", "Type", "Alt"]);

ok($err, "Chrom, Pos, Ref required");

done_testing();