use Test::More;
use Seq;
use DDP;
use lib 't/lib';

my $test_db_dir = Path::Tiny->tempdir();

my %baseArgs = (
  database_dir     => 't/tracks/gene/db/raw',
  input_file       => 'foo',
  config           => 't/tracks/gene/db/raw/config.yaml',
  output_file_base => 'bar',
  tracks           => {
    tracks => [
      {
        name        => 'ref',
        type        => 'reference',
        assembly    => 'hg19',
        chromosomes => ['chr1'],
        files_dir   => 'fglla'
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
