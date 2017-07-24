use 5.10.0;
use strict;
use warnings;

use lib '../';
# Takes a CADD file and makes it into a bed-like file, retaining the property
# That each base has 3 (or 4 for ambiguous) lines
package Utils::CaddToBed;

our $VERSION = '0.001';

use Mouse 2;
use namespace::autoclean;
use Path::Tiny qw/path/;

use Seq::Tracks::Build::LocalFilesPaths;

# # _localFilesDir, _decodedConfig, compress, _wantedTrack, _setConfig, and logPath
extends 'Utils::Base';

# ########## Arguments accepted ##############
# Expects tab delimited file; not allowing to be set because it probably won't ever be anything
# other than tab, and because split('\'t') is faster
# has delimiter => (is => 'ro', lazy => 1, default => "\t");

sub BUILD {
  my $self = shift;

  my $localFilesHandler = Seq::Tracks::Build::LocalFilesPaths->new();

  my $localFilesAref = $localFilesHandler->makeAbsolutePaths($self->_decodedConfig->{files_dir},
    $self->_wantedTrack->{name}, $self->_wantedTrack->{local_files});

  if (@$localFilesAref != 1) {
    $self->log('fatal', "Expect a single cadd file, found " . (scalar @$localFilesAref) );
  }

  $self->{_localFile} = $localFilesAref->[0];
}

# TODO: error check opening of file handles, write tests
sub go {
  my $self = shift;

  my %wantedChrs = map { $_ => 1 } @{ $self->_decodedConfig->{chromosomes} };

  my $inFilePath = $self->{_localFile};

  if(! -e $inFilePath) {
    $self->log('fatal', "input file path $inFilePath doesn't exist");
    return;
  }
  # Store output handles by chromosome, so we can write even if input file
  # out of order
  my %outFhs;
  my %skippedBecauseExists;

  # We'll update this list of files in the config file
  $self->_wantedTrack->{local_files} = [];

  my $inFh = $self->get_read_fh($inFilePath);

  $self->log('info', "Reading $inFilePath");

  my $versionLine = <$inFh>;
  chomp $versionLine;

  $self->log('info', "Cadd version line: $versionLine");

  my $headerLine = <$inFh>;
  chomp $headerLine;

  $self->log('info', "Cadd header line: $headerLine");

  my @headerFields = split('\t', $headerLine);

  # CADD seems to be 1-based, this is not documented however.
  my $based = 1;

  my $outPathBase = path($inFilePath)->basename();

  my $outExt = 'bed'  . ( $self->compress ? '.gz' : substr($outPathBase,
    rindex($outPathBase, '.') ) );

  $outPathBase = substr($outPathBase , 0, rindex($outPathBase , '.') );

  my $outPath = path($self->_localFilesDir)->child("$outPathBase.$outExt")->stringify();

  if(-e $outPath && !$self->overwrite) {
    $self->log('fatal', "File $outPath exists, and overwrite is not set");
    return;
  }

  my $outFh = $self->get_write_fh($outPath);

  $self->log('info', "Writing to $outPath");

  say $outFh $versionLine;
  say $outFh join("\t", 'chrom', 'chromStart', 'chromEnd', @headerFields[2 .. $#headerFields]);

  while(my $l = $inFh->getline() ) {
    chomp $l;

    my @line = split('\t', $l);

    # The part that actually has the id, ex: in chrX "X" is the id
    my $chrIdPart;
    # Get the chromosome
    # It could be stored as a number/single character or "chr"
    # Grab the chr part, and normalize it to our case format (chr)
    if($line[0] =~ /chr/i) {
      $chrIdPart = substr($line[0], 3);
    } else {
      $chrIdPart = $line[0];
    }

    # Don't forget to convert NCBI to UCSC-style mitochondral chr name
    if($chrIdPart eq 'MT') {
      $chrIdPart = 'M';
    }

    my $chr = "chr$chrIdPart";

    if(!exists $wantedChrs{$chr}) {
      $self->log('warn', "Chromosome $chr not recognized (from $chrIdPart), skipping: $l");
      next;
    }

    my $start = $line[1] - $based;
    my $end = $start + 1;
    say $outFh join("\t", $chr, $start, $end, @line[2 .. $#line]);
  }

  $self->_wantedTrack->{local_files} = [$outPath];

  $self->_wantedTrack->{caddToBed_date} = $self->_dateOfRun;

  $self->_backupAndWriteConfig();
}

__PACKAGE__->meta->make_immutable;
1;
