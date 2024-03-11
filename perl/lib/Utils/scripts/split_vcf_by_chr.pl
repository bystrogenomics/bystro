#!/usr/bin/perl

use strict;
use warnings;
use File::Temp qw/ :POSIX /;
use Path::Tiny;
use DDP;

# Check for command line argument
die "Usage: $0 <VCF_file>\n" unless @ARGV == 1;

# Variables
my $vcf_file   = $ARGV[0];
my $use_bgzip  = system("command -v bgzip > /dev/null") == 0;
my $is_gzipped = `file --mime-type "$vcf_file"` =~ /gzip$/;
my $read_cmd =
  $is_gzipped ? $use_bgzip ? "bgzip --threads 32 -d -c" : "gzip -d -c" : "cat";
my $write_cmd =
  $is_gzipped ? $use_bgzip ? "bgzip --threads 32 -c" : "gzip -c" : "cat";
my $num_cores = `nproc`;

my $vcf_file_dir      = path($vcf_file)->dirname;
my $vcf_file_basename = path($vcf_file)->basename( '.gz', '.bgz' );

# Read the VCF header
open my $fh, "-|", "$read_cmd $vcf_file" or die "Cannot open file: $!";
my ( @header_before_contig, $final_header_line );
my $header_count = 0;
while (<$fh>) {
  if (/^#/) {
    $header_count += 1;
    if (/^##contig=/) {
      next;
    }

    if (/^#CHROM/) {
      $final_header_line = $_;
      last;
    }

    push @header_before_contig, $_;
  }
}
close $fh;

# Sort the VCF file by chromosome
my $cmd = "$read_cmd $vcf_file";
open my $sorted_vcf_fh, "-|", $cmd or die "Cannot open file: $!";

say STDERR "\nREAD COMMAND: $cmd\n";
# Process the sorted file
my $prev_chr_file;
my %chr_fhs    = ();
my $line_count = 0;
while ( my $line = <$sorted_vcf_fh> ) {
  $line_count += 1;

  if ( $line_count <= $header_count ) {
    say STDERR "Skipping header line: $line";
    next;
  }

  if ( $line_count == $header_count + 1 ) {
    say STDERR "Processing first line: $line";
  }

  my $tab_index = index( $line, "\t" );
  my $chr       = substr( $line, 0, $tab_index );

  if ( !$chr ) {
    say STDERR "Couldn't find chromosome on line: $line";
    exit 1;
  }

  if ( !$chr_fhs{$chr} ) {
    my $chr_file = path($vcf_file_dir)
      ->child( "$vcf_file_basename.$chr.vcf" . ( $is_gzipped ? ".gz" : "" ) )->stringify;
    open my $fh, "|-", "$write_cmd > $chr_file" or die "Cannot open file: $!";

    say STDERR "Opened: $chr_file";

    my $final_header =
      join( "", @header_before_contig ) . "##contig=<ID=$chr>\n" . $final_header_line;
    print $fh $final_header;

    $chr_fhs{$chr} = $fh;
  }

  print { $chr_fhs{$chr} } $line;
}
close $sorted_vcf_fh;

say STDERR "Processed $line_count lines";

for my $chr ( keys %chr_fhs ) {
  say STDERR "Closing: $chr";
  close $chr_fhs{$chr};
}

print "VCF file split by chromosomes completed.\n";
