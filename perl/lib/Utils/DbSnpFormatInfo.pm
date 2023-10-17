#!/usr/bin/perl
use 5.10.0;
use strict;
use warnings;
use File::Basename;

# Takes a CADD file and makes it into a bed-like file, retaining the property
# That each base has 3 (or 4 for ambiguous) lines
package Utils::ReformatDbSnpVCF;

our $VERSION = '0.001';

use Mouse 2;
use namespace::autoclean;
use Path::Tiny qw/path/;

use Seq::Tracks::Build::LocalFilesPaths;

# Exports: _localFilesDir, _decodedConfig, compress, _wantedTrack, _setConfig, logPath, use_absolute_path
extends 'Utils::Base';

# ########## Arguments accepted ##############
# Expects tab delimited file; not allowing to be set because it probably won't ever be anything
# other than tab, and because split('\'t') is faster
# has delimiter => (is => 'ro', lazy => 1, default => "\t");

sub BUILD {
  my $self = shift;

  my $localFilesHandler = Seq::Tracks::Build::LocalFilesPaths->new();

  my $localFilesAref = $localFilesHandler->makeAbsolutePaths(
    $self->_decodedConfig->{files_dir},
    $self->_wantedTrack->{name},
    $self->_wantedTrack->{local_files}
  );

  $self->{_localFiles} = $localFilesAref;
}

# TODO: error check opening of file handles, write tests
sub go {
  my $self = shift;

  my @output_paths;

  for my $input_vcf ( @{ $self->{_localFiles} } ) {
    if ( !-e $input_vcf ) {
      $self->log( 'fatal', "input file path $input_vcf doesn't exist" );
      return;
    }

    my $base_name = basename($input_vcf);
    $base_name =~ s/\.[^.]+$//;     # Remove last file extension (if present)
    $base_name =~ s/\.[^.]+$//;     # Remove another file extension if it's something like .vcf.gz

    my $output_vcf_data = $base_name . "_vcf_data.vcf.gz";
    my $output_vcf_header = $base_name . "_vcf_header.vcf.gz";
    my $output_vcf = $base_name . "_processed.vcf.gz";

    my $in_fh = $self->getReadFh($input_vcf);

    $self->log( 'info', "Reading $input_vcf" );

    my $output_data_path = path( $self->_localFilesDir )->child("$output_vcf_data")->stringify();

    if ( -e $output_data_path && !$self->overwrite ) {
      $self->log( 'fatal', "Temp file $output_data_path exists, and overwrite is not set" );
      return;
    }

    my $output_data_fh = $self->getWriteFh($output_data_path);

    $self->log( 'info', "Writing to $output_data_path" );

    # Store populations seen across the VCF
    my %populations;

    my @header_lines;
    while (<$in_fh>) {
        chomp;
        
        # If it's a header line
        if (/^#/) {
            push @header_lines, $_;
            next;
        }

        # If it's an INFO line
        if (/FREQ=/) {
            my @info_fields = split(/;/, $_);
            my @new_info_fields;
            my %freqs;

            foreach my $info (@info_fields) {
                if ($info =~ /FREQ=(.+)/) {
                    my $freq_data = $1;
                    my @pops = split(/\|/, $freq_data);
                    
                    foreach my $pop (@pops) {
                        if ($pop =~ /([^:]+):(.+)/) {
                            my $pop_name = $1;
                            my @freq_vals = split(/,/, $2);
                            shift @freq_vals; # Remove the reference allele freq
                            $freqs{$pop_name} = join(",", @freq_vals);
                            $populations{$pop_name} = 1;
                        }
                    }
                } else {
                    push @new_info_fields, $info; # Keep the existing INFO fields
                }
            }
            
            # Append the new frequency data to the INFO field
            foreach my $pop_name (keys %freqs) {
                push @new_info_fields, "$pop_name=$freqs{$pop_name}";
            }

            say $output_data_fh join(";", @new_info_fields);
        }
    }

    close($in_fh);
    close($output_data_fh);
  }

  $self->_wantedTrack->{local_files} = [$outPath];

  $self->_backupAndWriteConfig();

  

  my $versionLine = <$inFh>;
  chomp $versionLine;

  $self->log( 'info', "Cadd version line: $versionLine" );

  my $headerLine = <$inFh>;
  chomp $headerLine;

  $self->log( 'info', "Cadd header line: $headerLine" );

  my @headerFields = split( '\t', $headerLine );

  # CADD seems to be 1-based, this is not documented however.
  my $based = 1;

  my $outPathBase = path($inFilePath)->basename();

  my $outExt = 'bed'
    . (
    $self->compress ? '.gz' : substr( $outPathBase, rindex( $outPathBase, '.' ) ) );

  $outPathBase = substr( $outPathBase, 0, rindex( $outPathBase, '.' ) );

  my $outPath =
    path( $self->_localFilesDir )->child("$outPathBase.$outExt")->stringify();

  if ( -e $outPath && !$self->overwrite ) {
    $self->log( 'fatal', "File $outPath exists, and overwrite is not set" );
    return;
  }

  my $outFh = $self->getWriteFh($outPath);

  $self->log( 'info', "Writing to $outPath" );

  say $outFh $versionLine;
  say $outFh join( "\t",
    'chrom', 'chromStart', 'chromEnd', @headerFields[ 2 .. $#headerFields ] );

  while ( my $l = $inFh->getline() ) {
    chomp $l;

    my @line = split( '\t', $l );

    # The part that actually has the id, ex: in chrX "X" is the id
    my $chrIdPart;
    # Get the chromosome
    # It could be stored as a number/single character or "chr"
    # Grab the chr part, and normalize it to our case format (chr)
    if ( $line[0] =~ /chr/i ) {
      $chrIdPart = substr( $line[0], 3 );
    }
    else {
      $chrIdPart = $line[0];
    }

    # Don't forget to convert NCBI to UCSC-style mitochondral chr name
    if ( $chrIdPart eq 'MT' ) {
      $chrIdPart = 'M';
    }

    my $chr = "chr$chrIdPart";

    if ( !exists $wantedChrs{$chr} ) {
      $self->log( 'warn',
        "Chromosome $chr not recognized (from $chrIdPart), skipping: $l" );
      next;
    }

    my $start = $line[1] - $based;
    my $end   = $start + 1;
    say $outFh join( "\t", $chr, $start, $end, @line[ 2 .. $#line ] );
  }

  $self->_wantedTrack->{local_files} = [$outPath];

  $self->_backupAndWriteConfig();
}

__PACKAGE__->meta->make_immutable;
1;


use strict;
use warnings;
use File::Basename;
use IO::Uncompress::Gunzip qw($GunzipError);
use DDP;

# Check for the correct number of command-line arguments
unless (@ARGV == 1) {
    die "Usage: $0 <input_vcf or input_vcf.gz> <output_vcf.gz>\n";
}

# Define the input and output filenames from command-line arguments
my ($input_vcf) = @ARGV;

my $base_name = basename($input_vcf);
$base_name =~ s/\.[^.]+$//;     # Remove last file extension (if present)
$base_name =~ s/\.[^.]+$//;     # Remove another file extension if it's something like .vcf.gz

my $output_vcf_data = $base_name . "_vcf_data.vcf.gz";
my $output_vcf_header = $base_name . "_vcf_header.vcf.gz";
my $output_vcf = $base_name . "_processed.vcf.gz";


# Declare variable to store VCF header
my $vcf_header = "";

# Store populations seen across the VCF
my %populations;

# Process the VCF

my $fh_in;
if ($input_vcf =~ /\.gz$/) {
    $fh_in = new IO::Uncompress::Gunzip($input_vcf) or die "Cannot open $input_vcf: $GunzipError";
} else {
    open($fh_in, "<", $input_vcf) or die "Cannot open $input_vcf: $!";
}

open(my $fh_out, "| gzip -c > $output_vcf_data") or die "Cannot open $output_vcf_data: $!";

my @header_lines;
while (<$fh_in>) {
    chomp;
    
    # If it's a header line
    if (/^#/) {
        push @header_lines, $_;
        next;
    }

    # my @fields = split(/\t/, $_);
    # If it's an INFO line
    if (/FREQ=/) {
        my @info_fields = split(/;/, $_);
        my @new_info_fields;
        my %freqs;

        foreach my $info (@info_fields) {
            if ($info =~ /FREQ=(.+)/) {
                my $freq_data = $1;
                my @pops = split(/\|/, $freq_data);
                
                foreach my $pop (@pops) {
                    if ($pop =~ /([^:]+):(.+)/) {
                        my $pop_name = $1;
                        my @freq_vals = split(/,/, $2);
                        shift @freq_vals; # Remove the reference allele freq
                        $freqs{$pop_name} = join(",", @freq_vals);
                        $populations{$pop_name} = 1;
                    }
                }
            } else {
                push @new_info_fields, $info; # Keep the existing INFO fields
            }
        }
        
        # Append the new frequency data to the INFO field
        foreach my $pop_name (keys %freqs) {
            push @new_info_fields, "$pop_name=$freqs{$pop_name}";
        }

        say $fh_out join(";", @new_info_fields);
    }
}

close($fh_in);
close($fh_out);

# Update the VCF header with new populations
my @pop_lines;
foreach my $pop (keys %populations) {
    push @pop_lines, "##INFO=<ID=$pop,Number=A,Type=Float,Description=\"Frequency for $pop\">";
}

splice(@header_lines, -1, 0, @pop_lines);

my $output_header_file = $output_vcf . ".vcf.gz";
open (my $header_fh, "| gzip -c > $output_vcf_header") or die "Cannot open $output_header_file: $!";
# Write the updated header and VCF to output
# open($fh_out, "| gzip -c > $output_vcf") or die "Cannot open $output_vcf: $!";
say $header_fh join("\n", @header_lines);
close($header_fh);

system("cat $output_vcf_header $output_vcf_data > $output_vcf") == 0 or die "Failed to concatenate files: $?";
system("rm $output_vcf_header $output_vcf_data") == 0 or die "Failed to remove temporary files: $?";

print "VCF processing complete.\n";
