#!/usr/bin/perl
use strict;
use warnings;
use Test::More;

use Path::Tiny;
use YAML::XS qw/DumpFile/;

use Utils::DbSnp2FormatInfo;

# create temp directories
my $db_dir   = Path::Tiny->tempdir();
my $raw_dir   = Path::Tiny->tempdir();

my $vcf_path = $raw_dir->child('test.vcf')->stringify;
my $expected_output_vcf_path = $raw_dir->child('dbSNP/test_processed.vcf')->stringify;

my $config = {
  'assembly'     => 'hg38',
  'chromosomes'  => [ 'chr1' ],
  'database_dir' => $db_dir->stringify,
  'files_dir'    => $raw_dir->stringify,
  'tracks'       => {
    'tracks' => [
      {
        'local_files'          => [$vcf_path],
        'name'          => 'dbSNP',
        'sorted'        => 1,
        'type'          => 'vcf',
        'utils' => [
            {
                'name' => 'DbSnp2FormatInfo'
            }
        ]
      }
    ]
  }
};

# write temporary config file
my $config_file = $raw_dir->child('filterCadd.yml');
DumpFile( $config_file, $config );

# Prepare a sample VCF for testing
my $vcf_data = <<'END_VCF';
##fileformat=VCFv4.1
##INFO=<ID=RS,Number=1,Type=String,Description="dbSNP ID">
##INFO=<ID=dbSNPBuildID,Number=1,Type=Integer,Description="dbSNP Build ID">
##INFO=<ID=SSR,Number=0,Type=Flag,Description="Variant is a short tandem repeat">
#CHROM  POS     ID      REF     ALT     QUAL    FILTER  INFO
NC_000001.11    10001   rs1570391677    T       A,C     .       .       RS=1570391677;dbSNPBuildID=154;SSR=0;PSEUDOGENEINFO=DDX11L1:100287102;VC=SNV;R5;GNO;FREQ=KOREAN:0.9891,0.0109,.|SGDP_PRJ:0,1,.|dbGaP_PopFreq:1,.,0
NC_000001.11    10002   rs1570391692    A       C       .       .       RS=1570391692;dbSNPBuildID=154;SSR=0;PSEUDOGENEINFO=DDX11L1:100287102;VC=SNV;R5;GNO;FREQ=KOREAN:0.9944,0.005597
END_VCF


# Write sample VCF to a temporary file
open my $fh, '>', $vcf_path or die "Could not open $vcf_path: $!";
print $fh $vcf_data;
close $fh;

# Initialize the utility and process the VCF
my $utility = Utils::DbSnp2FormatInfo->new(
  {
    config     => $config_file,
    name       => 'dbSNP',
    utilName   => 'DbSnp2FormatInfo'
  }
);

$utility->go($vcf_path);

# Check that the processed file exists and is correctly formatted
ok(-e $expected_output_vcf_path, "Processed VCF file exists");

# Read the processed file to check the INFO field
$fh = path($expected_output_vcf_path)->openr;

ok(<$fh> == "##fileformat=VCFv4.1", 'VCF fileformat is correctly processed');
ok(<$fh> == "##INFO=<ID=RS,Number=1,Type=String,Description=\"dbSNP ID\">", 'RS population is correctly processed');
ok(<$fh> == "##INFO=<ID=dbSNPBuildID,Number=1,Type=Integer,Description=\"dbSNP Build ID\">", 'dbSNPBuildID population is correctly processed');
ok(<$fh> == "##INFO=<ID=SSR,Number=0,Type=Flag,Description=\"Variant is a short tandem repeat\">", 'SSR population is correctly processed');
ok(<$fh> == "##INFO=<ID=KOREAN,Number=A,Type=Float,Description=\"Frequency for KOREAN\">", 'KOREAN population is correctly processed');
ok(<$fh> == "##INFO=<ID=SGDP_PRJ,Number=A,Type=Float,Description=\"Frequency for SGDP_PRJ\">", 'SGDP_PRJ population is correctly processed');
ok(<$fh> == "##INFO=<ID=dbGaP_PopFreq,Number=A,Type=Float,Description=\"Frequency for dbGaP_PopFreq\">", 'dbGaP_PopFreq population is correctly processed');
ok(<$fh> == "#CHROM  POS     ID      REF     ALT     QUAL    FILTER  INFO");

ok(<$fh> == "NC_000001.11    10001   rs1570391677    T       A,C     .       .       RS=1570391677;dbSNPBuildID=154;SSR=0;PSEUDOGENEINFO=DDX11L1:100287102;VC=SNV;R5;GNO;KOREAN=0.0109,.;SGDP_PRJ=0,.;dbGaP_PopFreq=.,0", '1st data row wiht KOREAN, SGDP_PRJ, dbGap freqs are correctly processed');
ok(<$fh> == "NC_000001.11    10002   rs1570391692    A       C       .       .       RS=1570391692;dbSNPBuildID=154;SSR=0;PSEUDOGENEINFO=DDX11L1:100287102;VC=SNV;R5;GNO;KOREAN=0.005597", '2nd data row with KOREAN freq is correctly processed');

done_testing();