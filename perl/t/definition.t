use Test::More;
use Seq;
use DDP;
use lib 't/lib';

my $test_db_dir = Path::Tiny->tempdir();

my %baseArgs = (
  database_dir     => 't/tracks/gene/db/raw',
  input_file       => 'foo',
  output_file_base => 'bar',
  tracks           => {
    tracks => [
      {
        name        => 'ref',
        type        => 'reference',
        assembly    => 'hg19',
        chromosomes => ['chr1'],
        files_dir   => 'fglla',
      }
    ]
  },
  fileProcessors => {}
);

sub test_dosageMatrixOut {
  my $object = Seq->new(%baseArgs);

  my $outputFilesInfo = $object->outputFilesInfo;

  ok(
    defined $outputFilesInfo->{dosageMatrixOutPath},
    'dosageMatrixOutPath should be defined'
  );
  like( $outputFilesInfo->{dosageMatrixOutPath},
    qr/dosage\.feather$/, 'dosageMatrixOutPath should end with dosage.feather' );
}

# Mock a minimal configuration for testing
sub mockConfig {
  my ( $type, $args ) = shift;
  return {
    input_file     => "/path/to/input_file",
    fileProcessors => {
      $type => {
        program  => "mockProgram",
        args     => "--arg1 --arg2 \%output\%",
        no_stdin => 0,                         # or 1, depending on the test case
      }
    },
    outputFilesInfo => { output => "/path/to/output_file", },
    _workingDir     => Path::Tiny->tempdir,
    # Define other necessary attributes or methods here
  };
}

sub test_preparePreprocessorProgram {
  my $object = Seq->new(
    %baseArgs,
    fileProcessors => {
      vcf => {
        args    => '--sample %sampleList% --dosageOutput %dosageMatrixOutPath%',
        program => 'mockProgram'
      }
    }
  );

  my ( $finalProgram, $errPath ) = $object->_preparePreprocessorProgram('vcf');

  like(
    $finalProgram,
    qr/--dosageOutput \S*bar\.dosage\.feather/,
    'Check --dosageOutput includes "bar.dosage.feather"'
  );
  like(
    $finalProgram,
    qr/--sample \S*bar\.sample_list/,
    'Check --sample includes "bar.sample_list"'
  );
  unlike( $finalProgram, qr/%sampleList%/, 'Command does not contain "%sampleList%"' );
  unlike( $finalProgram, qr/%dosageMatrixOutPath%/,
    'Command does not contain "%dosageMatrixOutPath%"' );

}

test_dosageMatrixOut();
test_preparePreprocessorProgram();

done_testing();
