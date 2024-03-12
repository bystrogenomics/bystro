#!/usr/bin/perl
use 5.10.0;
use strict;
use warnings;

# Take a DbSNP 2 VCF file, and for each row, split the INFO field's FREQ data into separate INFO fields for each population
# NOTE: dbSNP VCF spec: https://www.ncbi.nlm.nih.gov/snp/docs/products/vcf/redesign/
# NOTE: that dbSNP uses a '.' to represent a missing value and  first allele is the reference, which is not the standard use.

package Utils::DbSnp2FormatInfo;

our $VERSION = '0.001';

use File::Basename qw/basename/;

use Mouse 2;
use namespace::autoclean;
use Path::Tiny qw/path/;

use Seq::Tracks::Build::LocalFilesPaths;

# Exports: _localFilesDir, _decodedConfig, compress, _wantedTrack, _setConfig, logPath, use_absolute_path
extends 'Utils::Base';

my $INFO_INDEX = 7;

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

sub _get_fh_paths {
  my ( $self, $input_vcf ) = @_;

  if ( !-e $input_vcf ) {
    $self->log( 'fatal', "input file path $input_vcf doesn't exist" );
    return;
  }
  say STDERR "input_vcf is $input_vcf\n";
  my ( $err, $isCompressed, $in_fh ) = $self->getReadFh($input_vcf);
  say STDERR "isCompressed is $isCompressed\n";
  $isCompressed ||= $self->compress;

  if ($err) {
    $self->log( 'fatal', $err );
    return;
  }

  my $base_name = basename($input_vcf);
  $base_name =~ s/\.[^.]+$//; # Remove last file extension (if present)
  $base_name
    =~ s/\.[^.]+$//; # Remove another file extension if it's something like .vcf.gz

  my $output_vcf_data = $base_name . "_vcf_data.vcf" . ( $isCompressed ? ".gz" : "" );
  my $output_vcf_header =
    $base_name . "_vcf_header.vcf" . ( $isCompressed ? ".gz" : "" );
  my $output_vcf = $base_name . "_processed.vcf" . ( $isCompressed ? ".gz" : "" );

  $self->log( 'info', "Reading $input_vcf" );

  my $output_header_path =
    path( $self->_localFilesDir )->child($output_vcf_header)->stringify();
  my $output_data_path =
    path( $self->_localFilesDir )->child($output_vcf_data)->stringify();
  my $output_path = path( $self->_localFilesDir )->child($output_vcf)->stringify();

  if ( ( -e $output_data_path || -e $output_header_path || -e $output_path )
    && !$self->overwrite )
  {
    $self->log( 'fatal',
      "Temp files $output_data_path, $output_header_path, or final output path $output_path exist, and overwrite is not set"
    );
    return;
  }

  return ( $in_fh, $output_data_path, $output_header_path, $output_path );
}

sub go {
  my $self = shift;

  my @output_paths;

  for my $input_vcf ( @{ $self->{_localFiles} } ) {
    my ( $in_fh, $output_data_path, $output_header_path, $output_path ) =
      $self->_get_fh_paths($input_vcf);
    say STDERR "output data path is $output_data_path\n";
    my $output_data_fh = $self->getWriteFh($output_data_path);

    $self->log( 'info', "Writing to $output_data_path" );

    my %populations;
    my @ordered_populations;

    my @header_lines;
    while (<$in_fh>) {
      chomp;

      # If it's a header line
      if (/^#/) {
        push @header_lines, $_;
        next;
      }

      my @fields = split( "\t", $_ );

      if ( !@fields ) {
        $self->log( "fatal", "No fields found in row: $_" );
        return;
      }

      my @info_fields = split( ";", $fields[$INFO_INDEX] );

      my @ordered_info_freqs;
      my %seen_info_pops;

      my $seen_freq = 0;
      foreach my $info (@info_fields) {
        if ( $info =~ /FREQ=(.+)/ ) {
          if ( $seen_freq == 1 ) {
            $self->log( "fatal", "FREQ seen twice in INFO field. Row: $_" );
            return;
          }

          $seen_freq = 1;

          my $freq_data = $1;
          my @pops      = split( /\|/, $freq_data );

          foreach my $pop (@pops) {
            if ( $pop =~ /([^:]+):(.+)/ ) {
              my $pop_name = $1;

              if ( exists $seen_info_pops{$pop_name} ) {
                self->log( "fatal", "Population $pop_name seen twice in INFO field. Row: $_" );
                return;
              }

              my @freq_vals = split( /,/, $2 );
              shift @freq_vals; # Remove the reference allele freq

              push @ordered_info_freqs, [ $pop_name, join( ",", @freq_vals ) ];

              if ( !exists $populations{$pop_name} ) {
                push @ordered_populations, $pop_name;
                $populations{$pop_name} = 1;
              }
            }
          }

          # Append the new frequency data to the INFO field
          my @new_info_fields;
          for my $res (@ordered_info_freqs) {
            my $name = $res->[0];
            my $freq = $res->[1];
            push @new_info_fields, "$name=$freq";
          }

          $info = join( ";", @new_info_fields );
        }
      }

      $fields[$INFO_INDEX] = join( ";", @info_fields );

      say $output_data_fh join( "\t", @fields );
    }

    close($in_fh);
    close($output_data_fh);

    # Update the VCF header with new populations
    my @pop_lines;
    foreach my $pop (@ordered_populations) {
      push @pop_lines,
        "##INFO=<ID=$pop,Number=A,Type=Float,Description=\"Frequency for $pop\">";
    }

    splice( @header_lines, -1, 0, @pop_lines );
    say STDERR "output header is $output_header_path\n";
    my $header_fh = $self->getWriteFh($output_header_path);

    # Write the updated header and VCF to output
    say $header_fh join( "\n", @header_lines );
    close($header_fh);

    system("cat $output_header_path $output_data_path > $output_path") == 0
      or die "Failed to concatenate files: $?";
    system("rm $output_header_path $output_data_path") == 0
      or die "Failed to remove temporary files: $?";

    $self->log( 'info', "$input_vcf processing complete" );

    push @output_paths, $output_path;
  }

  $self->_wantedTrack->{local_files} = \@output_paths;

  $self->_backupAndWriteConfig();
}

__PACKAGE__->meta->make_immutable;
1;
