#!/usr/bin/env perl

use strict;
use warnings;
use YAML::Tiny;
use IO::Uncompress::Gunzip qw(gunzip $GunzipError);
use IO::Compress::Gzip     qw(gzip $GzipError);
use File::Temp             qw(tempfile);

use strict;
use DDP;
use warnings;
use Test::More tests => 4; # Adjust the number of tests based on your test cases
require 'Utils/scripts/extract_gnomad_an_af_nhomalt.pl'
  ;                        # Replace with the name of your module

# Test for read_vcf_header with a non-gzipped VCF file
sub test_read_vcf_header_non_gzipped {
  my $header = read_vcf_header("path/to/non_gzipped_sample.vcf");
  ok( $header =~ /#CHROM/, "Header read correctly for non-gzipped file" );
}

# Test for read_vcf_header with a gzipped VCF file
sub test_read_vcf_header_gzipped {
  my $header = read_vcf_header("path/to/gzipped_sample.vcf.gz");
  ok( $header =~ /#CHROM/, "Header read correctly for gzipped file" );
}

# Test for parse_vcf_header
sub test_parse_vcf_header {
  my $header = "##INFO=<ID=AN, ...>\n##INFO=<ID=AF, ...>";
  my @types  = parse_vcf_header($header);
  p @types;
  is( scalar @types, 2, "Two types extracted" );
  ok( grep( /AN/, @types ), "AN type extracted" );
  ok( grep( /AF/, @types ), "AF type extracted" );
}

# Running the tests
test_parse_vcf_header();

# Helper function to create a temporary VCF file
sub create_temp_vcf {
  my ( $content, $is_gzipped ) = @_;

  my ( $fh, $filename ) =
    tempfile( SUFFIX => $is_gzipped ? '.vcf.gz' : '.vcf', UNLINK => 1 );

  if ($is_gzipped) {
    gzip \$content => $filename or die "gzip failed: $GzipError\n";
  }
  else {
    print $fh $content;
  }

  close $fh;

  return $filename;
}

# Example usage
my $vcf_content =
    "##fileformat=VCFv4.2\n"
  . "##INFO=<ID=AN,Number=1,Type=Integer,Description=\"Total number of alleles in called genotypes\">\n"
  . "##INFO=<ID=AF,Number=A,Type=Float,Description=\"Allele frequency\">\n"
  . "##INFO=<ID=control_AN,Number=A,Type=Float,Description=\"Allele frequency\">\n"
  . "##INFO=<ID=control_AF,Number=A,Type=Float,Description=\"Allele frequency\">\n"
  . "##INFO=<ID=control_AN_nfe,Number=A,Type=Float,Description=\"Allele frequency\">\n"
  . "##INFO=<ID=control_AF_nfe,Number=A,Type=Float,Description=\"Allele frequency\">\n"
  . "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n";

my $temp_vcf    = create_temp_vcf( $vcf_content, 0 ); # Create non-gzipped VCF
my $temp_vcf_gz = create_temp_vcf( $vcf_content, 1 ); # Create gzipped VCF

# Test with temporary files
my @expected = (
  "- AN: number",
  "- AF: number",
  "- control_AN: number",
  "- control_AF: number",
  "- control_AN_nfe: number",
  "- control_AF_nfe: number"
);
my @types = parse_vcf_header( read_vcf_header($temp_vcf) );
is_deeply( \@types, \@expected, "AN type extracted" );

@types = parse_vcf_header( read_vcf_header($temp_vcf_gz) );
is_deeply( \@types, \@expected, "AN type extracted from gzipped file" );
