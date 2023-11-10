#!/usr/bin/env perl

use strict;
use warnings;
use YAML::Tiny;
use IO::Uncompress::Gunzip qw(gunzip $GunzipError);

# Function to read the VCF header from a possibly gzipped file
sub read_vcf_header {
    my ($filename) = @_;
    my $header = '';

    # Check if file is gzipped
    if ( $filename =~ /\.gz|\.bgz$/ ) {
        my $z = IO::Uncompress::Gunzip->new($filename)
          or die "gunzip failed: $GunzipError\n";
        while (<$z>) {
            last          if /^#CHROM/;    # Stop at the column header line
            $header .= $_ if /^##/;
        }
        close $z;
    }
    else {
        open( my $fh, '<', $filename )
          or die "Could not open file '$filename' $!";
        while (<$fh>) {
            last          if /^#CHROM/;    # Stop at the column header line
            $header .= $_ if /^##/;
        }
        close $fh;
    }

    return $header;
}

# Function to parse the VCF header and extract the feature types
sub parse_vcf_header {
    my ($header) = @_;
    my @types;

    foreach my $line ( split /\n/, $header ) {
        if ( $line =~ /^##INFO=<ID=([^,]+(AN|AF|nhomalt)[^,]+)/ ) {
            push @types, "- $1: number";
        }
    }

    return @types;
}

my @types = parse_vcf_header( read_vcf_header( $ARGV[0] ) );

for my $type (@types) {
    print "$type\n";
}
