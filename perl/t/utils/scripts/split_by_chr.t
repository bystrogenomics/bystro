use 5.10.0;

use strict;
use warnings;

use File::Spec;
use Test::More; # Example, adjust based on actual tests

use Cwd 'abs_path';
use File::Basename;

use Path::Tiny;

# Determine the full path to the directory of the current script (test_my_script.pl)
my $test_script_dir = path($0)->absolute->parent;

# Calculate the path back to the 'bystro/' directory
my $bystro_dir = $test_script_dir->parent->parent->parent;

# Get the script path using Path::Tiny, which is in lib/Utils/scripts
my $script_path =
  $bystro_dir->child( 'lib', 'Utils', 'scripts', 'split_vcf_by_chr.pl' );

my $vcf_file_to_split = $test_script_dir->child('vcf_example.vcf');

my $split_command = "perl " . $script_path->stringify . " " . $vcf_file_to_split;
system($split_command) == 0 or die "Failed to execute $split_command: $!";

my @expected_chromosomes = ( '1', '2', '3' );
my $counts_of_entries    = {
  '1' => 3,
  '2' => 2,
  '3' => 1
};

# Verify each output file
my @expected_header_without_contig = (
  '##fileformat=VCFv4.2',
  "##INFO=<ID=AF,Number=A,Type=Float,Description=\"Allele Frequency\">"
);
foreach my $chrom (@expected_chromosomes) {
  my $output_file = $vcf_file_to_split->parent->child("vcf_example.vcf.$chrom.vcf");
  my @expected_header_with_contig = @expected_header_without_contig;
  push( @expected_header_with_contig, "##contig=<ID=$chrom>" );
  push( @expected_header_with_contig, "#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO" );

  ok( -e $output_file, "$output_file exists" );

  # Open the file and verify its contents
  # This is a basic check; you'll need to adjust it based on your expected data
  open( my $fh, '<', $output_file )
    or die "Could not open file '$output_file' $!";
  my $header_tested;
  my @headers;

  my $entry_count = 0;
  while ( my $line = <$fh> ) {
    chomp $line;

    # Accumulate headers
    if ( $line =~ /^#/ ) {
      push( @headers, $line );
      next;
    }

    if ( !$header_tested ) {
      is_deeply(
        \@headers,
        \@expected_header_with_contig,
        "Header for $chrom matches expected"
      );
      $header_tested = 1;
    }

    my ($chrom_from_file) = split( /\t/, $line );
    is( $chrom_from_file, $chrom, "Line chromosome matches expected: $chrom" );

    $entry_count += 1;
  }

  is(
    $entry_count,
    $counts_of_entries->{$chrom},
    "Entry count for $chrom matches expected"
  );
  close $fh;

  # Clean up file
  unlink($output_file);
}

done_testing();
