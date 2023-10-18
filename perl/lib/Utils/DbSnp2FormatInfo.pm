#!/usr/bin/perl
use 5.10.0;
use strict;
use warnings;

# Take a DbSNP 2 VCF file, and for each row, split the INFO field's FREQ data into separate INFO fields for each population
package Utils::DbSnp2FormatInfo;

our $VERSION = '0.001';

use File::Basename qw/basename/;

use Mouse 2;
use namespace::autoclean;
use Path::Tiny qw/path/;

use Seq::Tracks::Build::LocalFilesPaths;

# Exports: _localFilesDir, _decodedConfig, compress, _wantedTrack, _setConfig, logPath, use_absolute_path
extends 'Utils::Base';

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

    my $output_header_path = path( $self->_localFilesDir )->child("$output_vcf_header")->stringify();
    my $output_data_path = path( $self->_localFilesDir )->child("$output_vcf_data")->stringify();
    my $output_path = path( $self->_localFilesDir )->child("$output_vcf")->stringify();

    if ( (-e $output_data_path || -e $output_header_path || -e $output_path) && !$self->overwrite ) {
      $self->log( 'fatal', "Temp files $output_data_path, $output_header_path, or final output path $output_path exist, and overwrite is not set" );
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

    # Update the VCF header with new populations
    my @pop_lines;
    foreach my $pop (keys %populations) {
        push @pop_lines, "##INFO=<ID=$pop,Number=A,Type=Float,Description=\"Frequency for $pop\">";
    }

    splice(@header_lines, -1, 0, @pop_lines);

    my $header_fh = $self->getWriteFh($output_header_path);

    # Write the updated header and VCF to output
    say $header_fh join("\n", @header_lines);
    close($header_fh);

    system("cat $output_header_path $output_data_path > $output_path") == 0 or die "Failed to concatenate files: $?";
    system("rm $output_header_path $output_data_path") == 0 or die "Failed to remove temporary files: $?";

    $self->log( 'info', "$input_vcf processing complete" );

    push @output_paths, $output_path;
  }

  $self->_wantedTrack->{local_files} = \@output_paths;

  $self->_backupAndWriteConfig();
}

__PACKAGE__->meta->make_immutable;
1;