#!/usr/bin/perl
use strict;
use warnings;
use Test::More;

use Path::Tiny;
use YAML::XS qw/DumpFile/;

use Utils::DbSnp2FormatInfo;

# create temp directories
my $db_dir  = Path::Tiny->tempdir();
my $raw_dir = Path::Tiny->tempdir();

my $vcf_path = $raw_dir->child('test.vcf')->stringify;
my $expected_output_vcf_path =
  $raw_dir->child('dbSNP/test_processed.vcf')->stringify;

my $config = {
  'assembly'     => 'hg38',
  'chromosomes'  => ['chr1'],
  'database_dir' => $db_dir->stringify,
  'files_dir'    => $raw_dir->stringify,
  'tracks'       => {
    'tracks' => [
      {
        'local_files' => [$vcf_path],
        'name'        => 'dbSNP',
        'sorted'      => 1,
        'type'        => 'vcf',
        'utils'       => [ { 'name' => 'DbSnp2FormatInfo', args => {compress => 0}} ]
      }
    ]
  }
};

# write temporary config file
my $config_file = $raw_dir->child('filterCadd.yml');
DumpFile( $config_file, $config );

# Prepare a sample VCF for testing
my $vcf_data = "##fileformat=VCFv4.1\n";
$vcf_data .= "##INFO=<ID=RS,Number=1,Type=String,Description=\"dbSNP ID\">\n";
$vcf_data .=
  "##INFO=<ID=dbSNPBuildID,Number=1,Type=Integer,Description=\"dbSNP Build ID\">\n";
$vcf_data .=
  "##INFO=<ID=SSR,Number=0,Type=Flag,Description=\"Variant is a short tandem repeat\">\n";
$vcf_data .= "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n";
$vcf_data .=
  "NC_000001.11\t10001\trs1570391677\tT\tA,C\t.\t.\tRS=1570391677;dbSNPBuildID=154;SSR=0;PSEUDOGENEINFO=DDX11L1:100287102;VC=SNV;R5;GNO;FREQ=KOREAN:0.9891,0.0109,.|SGDP_PRJ:0,1,.|dbGaP_PopFreq:1,.,0\n";
$vcf_data .=
  "NC_000001.11\t10002\trs1570391692\tA\tC\t.\t.\tRS=1570391692;dbSNPBuildID=154;SSR=0;PSEUDOGENEINFO=DDX11L1:100287102;VC=SNV;R5;GNO;FREQ=KOREAN:0.9944,0.005597\n";
# What happens if we get a field after the freq field?
$vcf_data .=
  "NC_000001.11\t10002\trs1570391692\tA\tC\t.\t.\tRS=1570391692;SSR=0;PSEUDOGENEINFO=DDX11L1:100287102;VC=SNV;R5;GNO;FREQ=SOMEOTHER:0.99,0.01;dbSNPBuildID=154";
# Write sample VCF to a temporary file
open my $fh, '>', $vcf_path or die "Could not open $vcf_path: $!";
say $fh $vcf_data;
close $fh;

# Initialize the utility and process the VCF
my $utility = Utils::DbSnp2FormatInfo->new(
  {
    config   => $config_file,
    name     => 'dbSNP',
    utilName => 'DbSnp2FormatInfo',
    compress => undef
  }
);

$utility->go($vcf_path);

# Check that the processed file exists and is correctly formatted
ok( -e $expected_output_vcf_path, "Processed VCF file exists" );

# Read the processed file to check the INFO field
$fh = path($expected_output_vcf_path)->openr;

my @lines = <$fh>;

ok( $lines[0] eq "##fileformat=VCFv4.1\n", 'VCF fileformat is correctly processed' );
ok( $lines[1] eq "##INFO=<ID=RS,Number=1,Type=String,Description=\"dbSNP ID\">\n",
  'RS population is correctly processed' );
ok(
  $lines[2] eq
    "##INFO=<ID=dbSNPBuildID,Number=1,Type=Integer,Description=\"dbSNP Build ID\">\n",
  'dbSNPBuildID population is correctly processed'
);
ok(
  $lines[3] eq
    "##INFO=<ID=SSR,Number=0,Type=Flag,Description=\"Variant is a short tandem repeat\">\n",
  'SSR population is correctly processed'
);
ok(
  $lines[4] eq
    "##INFO=<ID=KOREAN,Number=A,Type=Float,Description=\"Frequency for KOREAN\">\n",
  'KOREAN population is correctly processed'
);
ok(
  $lines[5] eq
    "##INFO=<ID=SGDP_PRJ,Number=A,Type=Float,Description=\"Frequency for SGDP_PRJ\">\n",
  'SGDP_PRJ population is correctly processed'
);
ok(
  $lines[6] eq
    "##INFO=<ID=dbGaP_PopFreq,Number=A,Type=Float,Description=\"Frequency for dbGaP_PopFreq\">\n",
  'dbGaP_PopFreq population is correctly processed'
);
ok(
  $lines[7] eq
    "##INFO=<ID=SOMEOTHER,Number=A,Type=Float,Description=\"Frequency for SOMEOTHER\">\n",
  'SOMEOTHER population is correctly processed'
);
ok( $lines[8] eq "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n" );

ok(
  $lines[9] eq
    "NC_000001.11\t10001\trs1570391677\tT\tA,C\t.\t.\tRS=1570391677;dbSNPBuildID=154;SSR=0;PSEUDOGENEINFO=DDX11L1:100287102;VC=SNV;R5;GNO;KOREAN=0.0109,.;SGDP_PRJ=1,.;dbGaP_PopFreq=.,0\n",
  '1st data row with KOREAN, SGDP_PRJ, dbGap freqs are correctly processed'
);
ok(
  $lines[10] eq
    "NC_000001.11\t10002\trs1570391692\tA\tC\t.\t.\tRS=1570391692;dbSNPBuildID=154;SSR=0;PSEUDOGENEINFO=DDX11L1:100287102;VC=SNV;R5;GNO;KOREAN=0.005597\n",
  '2nd data row with KOREAN freq is correctly processed'
);
ok(
  $lines[11] eq
    "NC_000001.11\t10002\trs1570391692\tA\tC\t.\t.\tRS=1570391692;SSR=0;PSEUDOGENEINFO=DDX11L1:100287102;VC=SNV;R5;GNO;SOMEOTHER=0.01;dbSNPBuildID=154\n",
  '2nd data row with SOMEOTHER freq is correctly processed'
);

done_testing();
