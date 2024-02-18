use 5.10.0;
use strict;
use warnings;
# TODO: Also support reading zipped files (right now only gzip files)
package Seq::Role::IO;

our $VERSION = '0.001';

# ABSTRACT: A moose role for all of our file handle needs
# VERSION

use Mouse::Role;

use PerlIO::utf8_strict;
use PerlIO::gzip;
use File::Which qw/which/;
use Sys::CpuAffinity;

use Path::Tiny;
use Try::Tiny;

use Scalar::Util qw/looks_like_number/;
with 'Seq::Role::Message';
# tried various ways of assigning this to an attrib, with the intention that
# one could change the taint checking characters allowed but this is the simpliest
# one that worked; wanted it precompiled to improve the speed of checking
my $taintCheckRegex = qr{\A([\+\,\.\-\=\:\/\t\|\s\w\d/]+)\z};

has taintCheckRegex => (
  is       => 'ro',
  lazy     => 1,
  init_arg => undef,
  default  => sub { $taintCheckRegex },
);

has delimiter => (
  is      => 'ro',
  lazy    => 1,
  default => '\t',
  writer  => '_setDelimiter',
);

my $tar  = which('tar');
my $gzip = which('bgzip') || which('pigz') || which('gzip');
my $lz4  = which('lz4');

# Without this, pigz -d -c issues many system calls (futex)
# Thanks to Meltdown, this slows decompression substantially
# Up to 1 core will be used solely for meltdown-related overhead
# So disable mutli-threading
# For compression, tradeoff still worth it
my $gzipDcmpArgs = '-d -c';
if ( $gzip =~ /pigz/ ) {
  $gzipDcmpArgs = "-p 1 $gzipDcmpArgs";
}
elsif ( $gzip =~ /bgzip/ ) {
  $gzipDcmpArgs = "--threads " . Sys::CpuAffinity::getNumCpus() . " $gzipDcmpArgs";
}

my $gzipCmpArgs = '-c';
if ( $gzip =~ /bgzip/ ) {
  $gzipCmpArgs = "--threads " . Sys::CpuAffinity::getNumCpus();
}

my $tarCompressedGzip = "$tar --use-compress-program=$gzip";
my $tarCompressedLZ4  = "$tar --use-compress-program=$lz4";

has gzip => (
  is       => 'ro',
  isa      => 'Str',
  init_arg => undef,
  lazy     => 1,
  default  => sub { $gzip }
);
has decompressArgs => (
  is       => 'ro',
  isa      => 'Str',
  init_arg => undef,
  lazy     => 1,
  default  => sub { $gzipDcmpArgs }
);

sub getReadArgs {
  my ( $self, $filePath ) = @_;

  my ( $remoteCpCmd, $remoteFileSizeCmd ) = $self->getRemoteProg($filePath);
  my $outerCommand = $self->getCompressProgWithArgs($filePath);

  if ($remoteCpCmd) {
    if ($outerCommand) {
      $outerCommand = "$remoteCpCmd | $outerCommand -";
    }
    else {
      $outerCommand = "$remoteCpCmd";
    }
  }
  elsif ($outerCommand) {
    $outerCommand = "$outerCommand $filePath";
  }

  return $outerCommand;
}
#@param {Path::Tiny} $file : the Path::Tiny object representing a single input file
#@param {Str} $errCode : what log level to use if we can't open the file
#@return file handle
sub getReadFh {
  my ( $self, $file, $errCode ) = @_;

  # By default, we'll return an error, rather than die-ing with log
  # We wont be able to catch pipe errors however
  if ( !$errCode ) {
    $errCode = 'error';
  }

  my $filePath;
  if ( ref $file eq 'Path::Tiny' ) {
    $filePath = $file->stringify;
  }
  else {
    $filePath = $file;
  }

  my $outerCommand = $self->getReadArgs($filePath);

  my ( $err, $fh );

  if ($outerCommand) {
    $err = $self->safeOpen( $fh, '-|', "$outerCommand", $errCode );
  }
  else {
    $err = $self->safeOpen( $fh, '<', $filePath, $errCode );
  }

  my $compressed = !!$outerCommand;

  return ( $err, $compressed, $fh );
}

sub getRemoteProg {
  my ( $self, $filePath ) = @_;

  if ( $filePath =~ /^s3:\/\// ) {
    return ( "aws s3 cp $filePath -", "" );
  }

  if ( $filePath =~ /^gs:\/\// ) {
    return ( "gsutil cp $filePath -", "" );
  }

  return "";
}

sub getInnerFileCommand {
  my ( $self, $filePath, $innerFile, $errCode ) = @_;

  if ( !$errCode ) {
    $errCode = 'error';
  }

  my $compressed =
       $innerFile =~ /[.]gz$/
    || $innerFile =~ /[.]bgz$/
    || $innerFile =~ /[.]zip$/
    || $filePath  =~ /[.]lz4$/;

  my $innerCommand;
  if ( $filePath =~ /[.]lz4$/ ) {
    $innerCommand = $compressed ? "\"$innerFile\" | $lz4 -d -c -" : "\"$innerFile\"";
  }
  else {
    $innerCommand =
      $compressed ? "\"$innerFile\" | $gzip $gzipDcmpArgs -" : "\"$innerFile\"";
  }

  # We do this because we have not built in error handling from opening streams

  my $err;
  my $command;
  my $outerCompressed;

  if ( $filePath =~ /[.]tar/ ) {
    $command = "$tar -O -xf - $innerCommand";
  }
  else {
    $err = "When inner file provided, must provde a parent file.tar or file.tar.gz";

    $self->log( $errCode, $err );

    return ( $err, undef, undef );
  }

  # If an innerFile is passed, we assume that $file is a path to a tarball

  return ( $err, $compressed, $command );
}

# Is the file a single compressed file
sub isCompressedSingle {
  my ( $self, $filePath ) = @_;

  my $basename = path($filePath)->basename();

  if ( $basename =~ /tar[.]gz$/ ) {
    return 0;
  }

  if ( $basename =~ /[.]gz$/ || $basename =~ /[.]bgz$/ || $basename =~ /[.]zip$/ ) {
    return "gzip";
  }

  if ( $basename =~ /[.]lz4$/ ) {
    return "lz4";
  }

  return "";
}

sub getCompressProgWithArgs {
  my ( $self, $filePath ) = @_;

  my $ext = $self->isCompressedSingle($filePath);

  if ( !$ext ) {
    return "";
  }
  if ( $ext eq 'gzip' ) {
    return "$gzip $gzipDcmpArgs";
  }
  if ( $ext eq 'lz4' ) {
    return "$lz4 -d -c";
  }
}

# TODO: return error if failed
sub getWriteFh {
  my ( $self, $file, $compress, $errCode ) = @_;

  # By default, we'll return an error, rather than die-ing with log
  # We wont be able to catch pipe errors however
  if ( !$errCode ) {
    $errCode = 'error';
  }

  my $err;

  if ( !$file ) {
    $err = 'get_fh() expected a filename';
    $self->log( $errCode, $err );

    return ( $err, undef );
  }

  my $fh;
  my $hasGz  = $file =~ /[.]gz$/ || $file =~ /[.]bgz$/ || $file =~ /[.]zip$/;
  my $hasLz4 = $file =~ /[.]lz4$/;
  if ( $hasGz || $hasLz4 || $compress ) {
    if ( $hasLz4 || ( $compress && $compress =~ /[.]lz4$/ ) ) {
      $err = $self->safeOpen( $fh, "|-", "$lz4 -c > $file", $errCode );
    }
    else {
      $err = $self->safeOpen( $fh, "|-", "$gzip $gzipCmpArgs > $file", $errCode );
    }

  }
  else {
    $err = $self->safeOpen( $fh, ">", $file, $errCode );
  }

  return ( $err, $fh );
}

# Allows user to return an error; dies with logging by default
sub safeOpen {
  #my ($self, $fh, $operator, $operand, $errCode) = @_;
  #    $_[0], $_[1], $_[2],   $_[3], $_[4]

  # In some cases, file attempting to be read may not have been flushed
  # Clearest case is Log::Fast
  my $err = $_[0]->safeSystem('sync');

  # Modifies $fh/$_[1] by reference
  if ( $err || !open( $_[1], $_[2], $_[3] ) ) {
    $err = $err || $!;

    #$self    #$errCode                      #$operand
    $_[0]->log( $_[4] || 'debug', "Couldn't open $_[3]: $err ($?)" );
    return $err;
  }

  return;
}

sub safeClose {
  my ( $self, $fh, $errCode ) = @_;

  my $err = $self->safeSystem('sync');

  if ($err) {
    $self->log( $errCode || 'debug', "Couldn't sync before close due to: $err" );
    return $err;
  }

  if ( !close($fh) ) {
    $self->log( $errCode || 'debug', "Couldn't close due to: $! ($?)" );
    return $!;
  }

  return;
}

sub getCleanFields {
  my ( $self, $line ) = @_;

  chomp $line;
  if ( $line =~ m/$taintCheckRegex/xm ) {
    my @out;

    push @out, split $self->delimiter, $1;

    return \@out;
  }

  return undef;
}

sub getLineEndings {
  return $/;
}

sub setLineEndings {
  my ( $self, $firstLine ) = @_;

  if ( $firstLine =~ /\r\n$/ ) {
    $/ = "\r\n";
  }
  elsif ( $firstLine =~ /\n$/ ) {
    $/ = "\n";
  }
  elsif ( $firstLine =~ /\015/ ) {
    # Match ^M (MacOS style line endings, which Excel outputs on Macs)
    $/ = "\015";
  }
  else {
    return "Cannot discern line endings: Not Mac, Unix, or Windows style";
  }

  return "";
}

sub checkDelimiter {
  my ( $self, $line ) = @_;

  if ( $line =~ /^\s*\S+\t\S+/ ) {
    return 1;
  }

  return 0;
}

sub safeSystem {
  my ( $self, $cmd, $errCode ) = @_;

  my $return = system($cmd);

  if ( $return > 0 ) {
    $self->log( $errCode || 'debug',
      "Failed to execute $cmd. Return: $return, due to: $! ($?)" );
    return $!;
  }

  return;
}

sub setDelimiter {
  my ( $self, $line ) = @_;

  if ( $line =~ /^\s*\S+\t\S+/ ) {
    $self->_setDelimiter('\t');
  }
  elsif ( $line =~ /^\s*\S+,\S+/ ) {
    $self->_setDelimiter(',');
  }
  else {
    return "Line is not tab or comma delimited";
  }

  return "";
}

sub makeTarballName {
  my ( $self, $baseName, $compress ) = @_;

  return $baseName . ( $compress ? '.tar.gz' : '.tar' );
}

# Assumes if ref's are passed for dir, baseName, or compressedName, they are path tiny
sub compressDirIntoTarball {
  my ( $self, $dir, $tarballName ) = @_;

  if ( !$tar ) {
    $self->log( 'warn', 'No tar program found' );
    return 'No tar program found';
  }

  if ( ref $dir ) {
    $dir = $dir->stringify;
  }

  if ( !$tarballName ) {
    $self->log( 'warn', 'must provide baseName or tarballName' );
    return 'Must provide baseName or tarballName';
  }

  if ( ref $tarballName ) {
    $tarballName = $tarballName->stringify;
  }

  $self->log( 'info', 'Compressing all output files' );

  my @files = glob $dir;

  if ( !@files ) {
    $self->log( 'warn', "Directory is empty" );
    return 'Directory is empty';
  }

  my $tarProg =
      $tarballName =~ /tar.gz$/
    ? $tarCompressedGzip
    : ( $tarballName =~ /tar.lz4$/ ? $tarCompressedLZ4 : "tar" );
  my $tarCommand = sprintf(
    "cd %s; $tarProg --exclude '.*' --exclude %s -cf %s * --remove-files",
    $dir,
    $tarballName, #and don't include our new compressed file in our tarball
    $tarballName, # the name of our tarball
  );

  $self->log( 'debug', "compress command: $tarCommand" );

  my $err = $self->safeSystem($tarCommand);

  return $err;
}

# returns chunk size in kbytes
sub getChunkSize {
  my ( $self, $filePath, $parts, $min, $max ) = @_;

  # If given 0
  $parts ||= 1;

  if ( !$min ) {
    $min = 512;
  }

  if ( !$max ) {
    $max = 32768;
  }

  my $size = path($filePath)->stat()->size;

  # Use 15x the size of the file as a heuristic
  # VCF files compress roughly this well
  $size *= 15;

  my $chunkSize = CORE::int( $size / ( $parts * 4096 ) );

  if ( $chunkSize < $min ) {
    return ( undef, $min );
  }

  # Cap to make sure memory usage doesn't grow uncontrollably
  if ( $chunkSize > $max ) {
    return ( undef, $max );
  }

  return ( undef, $chunkSize );
}

no Mouse::Role;

1;
