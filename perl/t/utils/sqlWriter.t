use 5.10.0;
use strict;
use warnings;

use Test::More;

use Path::Tiny;

use Utils::SqlWriter;

my $out_dir = Path::Tiny->tempdir();
my $db      = 'hg19';

my %config = (
  sql => "SELECT r.*, (SELECT GROUP_CONCAT(DISTINCT(NULLIF(x.kgID, '')) SEPARATOR
              ';') FROM kgXref x WHERE x.refseq=r.name) AS kgID, (SELECT GROUP_CONCAT(DISTINCT(NULLIF(x.description,
              '')) SEPARATOR ';') FROM kgXref x WHERE x.refseq=r.name) AS description,
              (SELECT GROUP_CONCAT(DISTINCT(NULLIF(e.value, '')) SEPARATOR ';') FROM knownToEnsembl
              e JOIN kgXref x ON x.kgID = e.name WHERE x.refseq = r.name) AS ensemblID,
              (SELECT GROUP_CONCAT(DISTINCT(NULLIF(x.tRnaName, '')) SEPARATOR ';') FROM
              kgXref x WHERE x.refseq=r.name) AS tRnaName, (SELECT GROUP_CONCAT(DISTINCT(NULLIF(x.spID,
              '')) SEPARATOR ';') FROM kgXref x WHERE x.refseq=r.name) AS spID, (SELECT
              GROUP_CONCAT(DISTINCT(NULLIF(x.spDisplayID, '')) SEPARATOR ';') FROM kgXref
              x WHERE x.refseq=r.name) AS spDisplayID, (SELECT GROUP_CONCAT(DISTINCT(NULLIF(x.protAcc,
              '')) SEPARATOR ';') FROM kgXref x WHERE x.refseq=r.name) AS protAcc, (SELECT
              GROUP_CONCAT(DISTINCT(NULLIF(x.mRNA, '')) SEPARATOR ';') FROM kgXref x WHERE
              x.refseq=r.name) AS mRNA, (SELECT GROUP_CONCAT(DISTINCT(NULLIF(x.rfamAcc,
              '')) SEPARATOR ';') FROM kgXref x WHERE x.refseq=r.name) AS rfamAcc FROM
              refGene r WHERE r.name='NM_019046' OR r.name='NM_001009943' OR r.name='NM_001009941';",
  connection => {
    database => $db,
    host     => 'genome-mysql.soe.ucsc.edu',
    user     => 'genome',
    port     => '3306'
  },
  outputDir => $out_dir->stringify,
  compress  => 0,
);

my $sqlWriter = Utils::SqlWriter->new( \%config );

$sqlWriter->go();

my $exp = $out_dir->child("$db.kgXref.fetch.txt")->stringify;

open( my $fh, '<', $exp );

my @stuff = <$fh>;

ok( @stuff == 4, "Ok, got the expected number of rows" );

chomp @stuff;

my @rows;

for my $r (@stuff) {
  push @rows, [ split '\t', $r ];
}

my @head = @{ $rows[0] };

# We expect
# [
#     [0]  "bin",
#     [1]  "name",
#     [2]  "chrom",
#     [3]  "strand",
#     [4]  "txStart",
#     [5]  "txEnd",
#     [6]  "cdsStart",
#     [7]  "cdsEnd",
#     [8]  "exonCount",
#     [9]  "exonStarts",
#     [10] "exonEnds",
#     [11] "score",
#     [12] "name2",
#     [13] "cdsStartStat",
#     [14] "cdsEndStat",
#     [15] "exonFrames",
#     [16] "kgID",
#     [17] "description",
#     [18] "ensemblID",
#     [19] "tRnaName",
#     [20] "spID",
#     [21] "spDisplayID",
#     [22] "protAcc",
#     [23] "mRNA",
#     [24] "rfamAcc"
# ]

ok( @head == 25, "The first line is a header" );

my $idx = 0;
for my $f (@head) {
  if ( $f eq 'name' ) {
    last;
  }

  $idx++;
}

my @tx  = sort { $a cmp $b } ( $rows[1][$idx], $rows[2][$idx], $rows[3][$idx] );
my @exp = sort { $a cmp $b } ( 'NM_019046', 'NM_001009943', 'NM_001009941' );

ok( join( "\t", @tx ) eq join( "\t", @exp ), "Find expected tx" );

$out_dir->remove_tree();

close $fh;

%config = (
  sql => "SELECT r.*, (SELECT GROUP_CONCAT(DISTINCT(NULLIF(x.kgID, '')) SEPARATOR
              ';') FROM kgXref x WHERE x.refseq=r.name) AS kgID, (SELECT GROUP_CONCAT(DISTINCT(NULLIF(x.description,
              '')) SEPARATOR ';') FROM kgXref x WHERE x.refseq=r.name) AS description,
              (SELECT GROUP_CONCAT(DISTINCT(NULLIF(e.value, '')) SEPARATOR ';') FROM knownToEnsembl
              e JOIN kgXref x ON x.kgID = e.name WHERE x.refseq = r.name) AS ensemblID,
              (SELECT GROUP_CONCAT(DISTINCT(NULLIF(x.tRnaName, '')) SEPARATOR ';') FROM
              kgXref x WHERE x.refseq=r.name) AS tRnaName, (SELECT GROUP_CONCAT(DISTINCT(NULLIF(x.spID,
              '')) SEPARATOR ';') FROM kgXref x WHERE x.refseq=r.name) AS spID, (SELECT
              GROUP_CONCAT(DISTINCT(NULLIF(x.spDisplayID, '')) SEPARATOR ';') FROM kgXref
              x WHERE x.refseq=r.name) AS spDisplayID, (SELECT GROUP_CONCAT(DISTINCT(NULLIF(x.protAcc,
              '')) SEPARATOR ';') FROM kgXref x WHERE x.refseq=r.name) AS protAcc, (SELECT
              GROUP_CONCAT(DISTINCT(NULLIF(x.mRNA, '')) SEPARATOR ';') FROM kgXref x WHERE
              x.refseq=r.name) AS mRNA, (SELECT GROUP_CONCAT(DISTINCT(NULLIF(x.rfamAcc,
              '')) SEPARATOR ';') FROM kgXref x WHERE x.refseq=r.name) AS rfamAcc FROM
              refGene r WHERE r.name='Ndjfalkjsdlkajf';",
  connection => {
    database => $db,
    host     => 'genome-mysql.soe.ucsc.edu',
    user     => 'genome',
    port     => '3306'
  },
  outputDir => $out_dir->stringify,
  compress  => 0,
);

$sqlWriter = Utils::SqlWriter->new( \%config );

$sqlWriter->go();

$exp = $out_dir->child("$db.kgXref.fetch.txt")->stringify;

ok( !-e $exp, "No file generated when empty query" );

done_testing();

1;
