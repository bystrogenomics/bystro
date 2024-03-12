package SeqTest;

use Mouse;
with 'Seq::Role::IO';

1;

use strict;
use Test::More;

use Sys::CpuAffinity;
use SeqTest;

subtest 'Decompress Command' => sub {
  my $seqTest =
    SeqTest->new( gzip => 'bgzip', lz4 => 'lz4', zip => 'zip', unzip => 'unzip' )
    ; # Instantiate the test class

  my $gzipFile = "file.gz";
  my $zipFile  = "file.zip";
  my $lz4File  = "file.lz4";

  is( $seqTest->getCompressionType($gzipFile),
    'gzip', 'Correctly identifies gzip files' );
  is( $seqTest->getCompressionType($zipFile), 'zip',
    'Correctly identifies zip files' );
  is( $seqTest->getCompressionType($lz4File), 'lz4',
    'Correctly identifies lz4 files' );

  my $gzipCmd = $seqTest->getDecompressProgWithArgs($gzipFile);
  my $threads = Sys::CpuAffinity::getNumCpus();
  is(
    $gzipCmd,
    "bgzip --threads $threads -d -c",
    'Correctly returns gzip decompression command'
  );

  my $gzFileCmd = $seqTest->getReadArgs($gzipFile);
  is(
    $gzFileCmd,
    "bgzip --threads $threads -d -c $gzipFile",
    'Correctly returns gzip decompression command'
  );

  my $zipCmd = $seqTest->getDecompressProgWithArgs($zipFile);
  is( $zipCmd, "unzip -p", 'Correctly returns zip decompression command' );

  my $zipFileCmd = $seqTest->getReadArgs($zipFile);
  is( $zipFileCmd, "unzip -p $zipFile",
    'Correctly returns zip decompression command' );

  my $lz4Cmd = $seqTest->getDecompressProgWithArgs($lz4File);
  is( $lz4Cmd, "lz4 -d -c", 'Correctly returns lz4 decompression command' );

  my $lz4FileCmd = $seqTest->getReadArgs($lz4File);
  is(
    $lz4FileCmd,
    "lz4 -d -c $lz4File",
    'Correctly returns lz4 decompression command'
  );

  $seqTest = SeqTest->new( gzip => 'pigz' ); # Instantiate the test class
  $gzipCmd = $seqTest->getDecompressProgWithArgs($gzipFile);
  is( $gzipCmd, "pigz -d -c", 'Correctly returns gzip decompression command' );

  $gzFileCmd = $seqTest->getReadArgs($gzipFile);
  is(
    $gzFileCmd,
    "pigz -d -c $gzipFile",
    'Correctly returns gzip decompression command'
  );

  $seqTest = SeqTest->new( gzip => 'gzip' ); # Instantiate the test class
  $gzipCmd = $seqTest->getDecompressProgWithArgs($gzipFile);
  is( $gzipCmd, "gzip -d -c", 'Correctly returns gzip decompression command' );

  $gzFileCmd = $seqTest->getReadArgs($gzipFile);
  is(
    $gzFileCmd,
    "gzip -d -c $gzipFile",
    'Correctly returns gzip decompression command'
  );
};

subtest 'Compress Command' => sub {
  my $seqTest =
    SeqTest->new( gzip => 'bgzip', lz4 => 'lz4', zip => 'zip', unzip => 'unzip' )
    ; # Instantiate the test class

  my $gzipFile = "file.gz";
  my $bzipFile = "file.bgz";
  my $zipFile  = "file.zip";
  my $lz4File  = "file.lz4";

  my $gzipCmd = $seqTest->getCompresssedWriteCmd($gzipFile);
  my $threads = Sys::CpuAffinity::getNumCpus();
  is(
    $gzipCmd,
    "bgzip --threads $threads > $gzipFile",
    'Correctly returns gzip compression command'
  );

  my $bzipCmd = $seqTest->getCompresssedWriteCmd($bzipFile);
  is(
    $bzipCmd,
    "bgzip --threads $threads > $bzipFile",
    'Correctly returns bgzip compression command'
  );

  my $zipCmd = $seqTest->getCompresssedWriteCmd($zipFile);
  is( $zipCmd, "zip -q > $zipFile", 'Correctly returns zip compression command' );

  my $lz4Cmd = $seqTest->getCompresssedWriteCmd($lz4File);
  is( $lz4Cmd, "lz4 > $lz4File", 'Correctly returns lz4 compression command' );

  $seqTest = SeqTest->new( gzip => 'pigz' ); # Instantiate the test class

  $gzipCmd = $seqTest->getCompresssedWriteCmd($gzipFile);
  is(
    $gzipCmd,
    "pigz -p $threads > $gzipFile",
    'Correctly returns gzip compression command'
  );

  $seqTest = SeqTest->new( gzip => 'gzip' ); # Instantiate the test class

  $gzipCmd = $seqTest->getCompresssedWriteCmd($gzipFile);
  is( $gzipCmd, "gzip  > $gzipFile", 'Correctly returns gzip compression command' );
};

subtest 'File Handling' => sub {
  # print(which('gzip'));
  my $gzip = which('gzip');

  my $seqTest = SeqTest->new( gzip => $gzip );

  my ( $err, $fh ) = $seqTest->getWriteFh(undef);
  is( $err, 'getWriteFh() expected a filename', 'No filename should return an error' );

  sub testExtensions {
    my $seqTest = shift;

    foreach my $ext (qw(.gz .bgz .zip .lz4 .txt)) {
      my ( $err, $fh ) = $seqTest->getWriteFh("file$ext");
      ok( defined $fh, "$ext extension should open a file handle with compression" );

      # Test that we can write
      my $content = "Sample text for compression and decompression tests";
      print $fh $content;
      close $fh;

      # Test that we can read
      ( $err, undef, $fh ) = $seqTest->getReadFh("file$ext");
      my $readContent = <$fh>;
      is( $readContent, $content, 'Can write and read from file handle' );

      if ( $ext eq '.gz' || $ext eq '.bgz' ) {
        # Test that a .gz file is a valid gzipped file
        my $gzipFile = "file$ext";

        # Test gzip file integrity
        # open gzip --test file handle
        my $gzip = $seqTest->gzip;
        open( my $gzipTest, '-|', "$gzip --test file$ext" ) or die $!;
        my $testResult = <$gzipTest>;
        close $gzipTest;
        is( $testResult, undef, 'Correctly identifies valid gzip file' );
      }

      if ( $ext eq '.zip' ) {
        # Test that a .zip file is a valid zip file
        my $zipFile = "file$ext";

        my $unzip = $seqTest->unzip;
        open( my $unzipTest, '-|', "$unzip -q -t file$ext" ) or die $!;
        my $testResult = <$unzipTest>;
        close $unzipTest;
        chomp $testResult;
        is(
          $testResult,
          "No errors detected in compressed data of $zipFile.",
          'Correctly identifies valid zip file'
        );
      }

      if ( $ext eq '.lz4' ) {
        my $lz4File = "file$ext";

        my $lz4 = $seqTest->lz4;
        open( my $lz4Test, '-|', "$lz4 -q -t file$ext" ) or die $!;
        my $testResult = <$lz4Test>;
        close $lz4Test;
        is( $testResult, undef, 'Correctly identifies valid lz4 file' );
      }
    }
  }

  testExtensions($seqTest);

  my $bgzip = which('bgzip');
  $seqTest = SeqTest->new( gzip => $bgzip );

  testExtensions($seqTest);

  my $pigz = which('pigz');
  $seqTest = SeqTest->new( gzip => $pigz );

  testExtensions($seqTest);
};

done_testing();
