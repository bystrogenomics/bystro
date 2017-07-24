#!/usr/bin/env perl

use 5.10.0;
use strict;
use warnings;
use lib './lib';
use Interface;
use Getopt::Long;
use DDP;

my $app = Interface->new_with_options();

$app->annotate;
=head1 NAME

annotate_snpfile.pl

=head1 DESCRIPTION

This program annotates an input file, using the Bystro database

=head1 VALID_FILES

1. PEMapper/PECaller .snp file (typically has .snp extension, but we accept any extension, as long as file is properly formatted (see below)

2. VCF file (typically has .vcf extension, but we accept any extension, as long as file is properly formatted (see below))

=head1 EXAMPLES

  annotate_snpfile.pl --in </path/to/input> --out </path/to/output> --config config/hg19.yml 

=head1 AUTHOR

Alex Kotlar <akotlar@emory.edu>
=head1 NAME

=head1 SEE ALSO

Seq Package

=cut
