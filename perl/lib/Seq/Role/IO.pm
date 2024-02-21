use 5.10.0;
use strict;
use warnings;

package Seq::Role::IO;

our $VERSION = '0.001';

use Mouse::Role;

use PerlIO::utf8_strict;
use PerlIO::gzip;
use File::Which qw/which/;
use Sys::CpuAffinity;

use Path::Tiny;
use Try::Tiny;

use Scalar::Util qw/looks_like_number/;
with 'Seq::Role::Message';

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

has cat => (
  is      => 'ro',
  lazy    => 1,
  default => sub {
    return which('cat');
  },
);

has gzip => (
  is      => 'ro',
  lazy    => 1,
  default => sub {
    return which('bgzip') || which('pigz') || which('gzip');
  },
);

has lz4 => (
  is      => 'ro',
  lazy    => 1,
  default => sub {
    return which('lz4');
  },
);

has gzipDcmpArgs => (
  is       => 'ro',
  lazy     => 1,
  init_arg => undef,
  default  => sub {
    my $self = shift;

    my $gzip = $self->gzip;

    if ( $gzip =~ /bgzip/ ) {
      return "--threads " . Sys::CpuAffinity::getNumCpus() . " -d -c";
    }

    return "-d -c"; # pigz decompression parallelism is maximized without -p
  },
);

has gzipCmpArgs => (
  is       => 'ro',
  lazy     => 1,
  init_arg => undef,
  default  => sub {
    my $self = shift;
    my $gzip = $self->gzip;

    if ( $gzip =~ /pigz/ ) {
      return "-p " . Sys::CpuAffinity::getNumCpus();
    }

    if ( $gzip =~ /bgzip/ ) {
      return "--threads " . Sys::CpuAffinity::getNumCpus();
    }

    return "";
  },
);

has tar => (
  is      => 'ro',
  lazy    => 1,
  default => sub {
    return which('tar');
  },
);

has unzip => (
  is      => 'ro',
  lazy    => 1,
  default => sub {
    return which('unzip');
  },
);

has zip => (
  is      => 'ro',
  lazy    => 1,
  default => sub {
    return which('zip');
  },
);

sub getReadArgs {
  my ( $self, $filePath ) = @_;

  my ( $remoteCpCmd, $remoteFileSizeCmd ) = $self->getRemoteProg($filePath);
  my $outerCommand = $self->getDecompressProgWithArgs($filePath);

  if ($remoteCpCmd) {
    if ( $outerCommand != $self->cat ) {
      # If we're piping to unzip, that won't work with standard unzip
      # For now raise an exception
      if ( index( $outerCommand, $self->unzip ) != -1 ) {
        die "Cannot pipe to unzip with remote files";
      }

      $outerCommand = "$remoteCpCmd | $outerCommand -";
    }
    else {
      $outerCommand = "$remoteCpCmd";
    }
  }
  else {
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
  my $compressed   = index( $outerCommand, $self->cat ) == -1;

  my ( $err, $fh );

  if ($outerCommand) {
    $err = $self->safeOpen( $fh, '-|', "$outerCommand", $errCode );
  }
  else {
    $err = $self->safeOpen( $fh, '<', $filePath, $errCode );
  }

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

# Is the file a single compressed file
sub getCompressionType {
  my ( $self, $filePath ) = @_;

  my $basename = path($filePath)->basename();

  if ( $basename =~ /[.]gz$/ || $basename =~ /[.]bgz$/ ) {
    return "gzip";
  }

  if ( $basename =~ /[.]zip$/ ) {
    return "zip";
  }

  if ( $basename =~ /[.]lz4$/ ) {
    return "lz4";
  }

  return "";
}

sub getDecompressProgWithArgs {
  my ( $self, $filePath ) = @_;

  my $ext = $self->getCompressionType($filePath);

  if ( !$ext ) {
    return $self->cat;
  }

  if ( $ext eq 'gzip' ) {
    return $self->gzip . " " . $self->gzipDcmpArgs;
  }

  if ( $ext eq 'zip' ) {
    return $self->unzip . " -p";
  }

  if ( $ext eq 'lz4' ) {
    return $self->lz4 . " -d -c";
  }
}

sub getWriteFh {
  my ( $self, $file, $errCode ) = @_;

  # By default, we'll return an error, rather than die-ing with log
  # We wont be able to catch pipe errors however
  if ( !$errCode ) {
    $errCode = 'error';
  }

  my $err;

  if ( !$file ) {
    $err = 'getWriteFh() expected a filename';
    $self->log( $errCode, $err );

    return ( $err, undef );
  }

  my $fh;

  ( $err, my $cmd ) = $self->getCompresssedWriteCmd($file);

  if ($err) {
    return ( $err, undef );
  }

  if ($cmd) {
    $err = $self->safeOpen( $fh, "|-", $cmd, $errCode );
  }
  else {
    $err = $self->safeOpen( $fh, ">", $file, $errCode );
  }

  return ( $err, $fh );
}

sub getCompresssedWriteCmd {
  my ( $self, $file ) = @_;

  my $hasGz  = $file =~ /[.]gz$/ || $file =~ /[.]bgz$/;
  my $hasZip = $file =~ /[.]zip$/;
  my $hasLz4 = $file =~ /[.]lz4$/;

  if ($hasLz4) {
    if ( !$self->lz4 ) {
      return (
        "lz4 requested but lz4 program not passed to Seq::Role::IO or not found in system",
        undef );
    }
    return ( undef, $self->lz4 . " > $file" );
  }

  if ($hasGz) {
    if ( !$self->gzip ) {
      return (
        "gzip requested but gzip-compatible program not passed to Seq::Role::IO or not found in system",
        undef
      );
    }

    return ( undef, $self->gzip . " " . $self->gzipCmpArgs . " > $file" );
  }

  if ($hasZip) {
    if ( !$self->zip ) {
      return (
        "zip requested but zip program not passed to Seq::Role::IO or not found in system",
        undef );
    }

    return ( undef, $self->zip . " -q > $file" );
  }

  return ( undef, undef );
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

  if ( !$self->tar ) {
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

  my $tar  = $self->tar;
  my $gzip = $self->gzip;
  my $lz4  = $self->lz4;

  my $tarProg =
    $tarballName =~ /tar.gz$/
    ? "$tar --use-compress-program=$gzip"
    : ( $tarballName =~ /tar.lz4$/ ? "$tar --use-compress-program=$lz4" : "tar" );
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

no Mouse::Role;

1;
