use 5.14.0;
use strict;
use warnings;

# Adds dbnsfp to refGene
package Utils::RefGeneXdbnsfp;

our $VERSION = '0.001';

use Mouse 2;
use namespace::autoclean;
use Path::Tiny qw/path/;
use Parallel::ForkManager;
use Seq::Role::IO;
use Seq::Output::Delimiters;
use Seq::Tracks::Build::LocalFilesPaths;

use List::Util qw/uniq/;

# Exports: _localFilesDir, _decodedConfig, compress, _wantedTrack, _setConfig, logPath, use_absolute_path
extends 'Utils::Base';

# ########## Arguments accepted ##############
# Expects tab delimited file; not allowing to be set because it probably won't ever be anything
# other than tab, and because split('\'t') is faster
# has delimiter => (is => 'ro', lazy => 1, default => "\t");
has geneFile => ( is => 'ro', isa => 'Str', required => 1 );

sub BUILD {
  my $self = shift;

  my $localFilesHandler = Seq::Tracks::Build::LocalFilesPaths->new();

  $self->{_localFiles} = $localFilesHandler->makeAbsolutePaths(
    $self->_decodedConfig->{files_dir},
    $self->_wantedTrack->{name},
    $self->_wantedTrack->{local_files}
  );

  if ( !@{ $self->{_localFiles} } ) {
    $self->log( 'fatal', "Require some local files" );
  }
}

# TODO: error check opening of file handles, write tests
sub go {
  my $self = shift;

  $self->log( 'info', 'Beginning RefGeneXdbnsfp' );
  # Store output handles by chromosome, so we can write even if input file
  # out of order
  my %outFhs;
  my %skippedBecauseExists;

  my $dbnsfpFh = $self->getReadFh( $self->geneFile );

  my $header = <$dbnsfpFh>;

  #appropriate chomp
  $self->setLineEndings($header);
  chomp $header;

  my @dbNSFPheaderFields = split '\t', $header;

  # Unfortunately, dbnsfp has many errors, for instance, NM_207007 being associated
  # with CCL4L1 (in neither hg19 or 38 is this true: SELECT * FROM refGene LEFT JOIN kgXref ON refGene.name = kgXref.refseq LEFT JOIN knownToEnsembl ON kgXref.kgID = knownToEnsembl.name WHERE refGene.name='NM_207007' ;)
  # So wel'll get odd duplicates;
  # A safer option is to lose transcript specificity, but use the unique list of genes
  my @geneNameCols = qw/Gene_name/;
  my @geneNameIdx;

  for my $col (@geneNameCols) {
    my $idx = 0;
    for my $dCol (@dbNSFPheaderFields) {
      if ( $dCol eq $col ) {
        push @geneNameIdx, $idx;
      }

      $idx++;
    }
  }

  my $delims   = Seq::Output::Delimiters->new();
  my $posDelim = $delims->positionDelimiter;
  my $ovrDelim = $delims->overlapDelimiter;
  my $valDelim = $delims->valueDelimiter;

  # namespace
  @dbNSFPheaderFields = map { 'dbnsfp.' . $_ } @dbNSFPheaderFields;
  push @dbNSFPheaderFields, 'dbnsfp.pubmedID';

  # unfortunately uses a period as a multi-value delimiter...
  my $funcIdx;

  my $i = -1;
  for my $field (@dbNSFPheaderFields) {
    $i++;
    if ( $field eq 'dbnsfp.Function_description' ) {
      $funcIdx = $i;
    }
  }

  my %dbNSFP;
  while (<$dbnsfpFh>) {
    #appropriate chomp based on line endings
    chomp;

    # Strip redundant words
    $_ =~ s/TISSUE SPECIFICITY:\s*|FUNCTION:\s*|DISEASE:\s*|PATHWAY:\s*//g;

    my @pmidMatch = $_ =~ m/PubMed:(\d+)/g;
    if (@pmidMatch) {
      @pmidMatch = uniq(@pmidMatch);
    }

    $_ =~ s/\{[^\}]+\}//g;

    # say "length : " . (scalar @fields);

    # Uniprot / dbnsfp annoyingly inserts a bunch of compound values
    # that aren't really meant to be split on
    # it would require negative lookbehind to correctly split them
    # While that isn't difficult in perl, it wastes performance
    # Replace such values with commas
    my @innerStuff = $_
      =~ m/(?<=[\(\[\{])([^\(\[\{\)\]\}]*[$posDelim$valDelim$ovrDelim\/]+[^\(\[\{\)\]\}]+)+(?=[\]\}\)])/g;

    for my $match (@innerStuff) {
      my $cp = $match;
      $cp =~ s/[$posDelim$valDelim$ovrDelim\/]/,/g;
      substr( $_, index( $_, $match ), length($match) ) = $cp;
    }

    $_ =~ s/[^\w\[\]\{\}\(\)\t\n\r]+(?=[^\w ])//g;

    my @fields = split '\t', $_;

    my $index = -1;
    for my $field (@fields) {
      $index++;

      my @unique;
      if ( $index == $funcIdx ) {
        @unique = uniq( split /[\.]/, $field );
      }
      else {
        # split on [;] more effective, will split in cases like ); which /;/ won't
        @unique = uniq( split /[;]/, $field );
      }

      my @out;

      my $index = -1;
      for my $f (@unique) {
        $f =~ s/^\s+//;
        $f =~ s/\s+$//;

        # shouldn't be necessary, just in case
        $f =~ s/\s*[^\w\[\]\{\}\(\)]+\s*$//;

        $f =~ s/[$posDelim$valDelim$ovrDelim\/]+/,/g;

        if ( defined $f && $f ne '' ) {
          push @out, $f;
        }
      }

      $field = @out ? join ";", @out : ".";
    }

    if (@pmidMatch) {
      push @fields, join( ';', @pmidMatch );
    }
    else {
      push @fields, '.';
    }

    if ( @fields != @dbNSFPheaderFields ) {
      $self->log( 'fatal', "WTF: $_" );
    }

    my $i = -1;
    for my $idx (@geneNameIdx) {
      $i++;

      my @vals = split ';', $fields[$idx];

      # sometimes dbNSFP gives duplicate values in the same string...
      my %seenThisLoop;
      for my $val (@vals) {
        if ( $val eq '.' || $val !~ /^\w+/ ) {
          $self->log( 'fatal', "WTF: missing gene?" );
        }

        $seenThisLoop{$val} = 1;

        if ( exists $dbNSFP{$val} ) {
          $self->log( 'fatal', "Duplicate entry found: $val, skipping : $_" );
          next;
        }

        $dbNSFP{$val} = \@fields;
      }
    }
  }

  # We'll update this list of files in the config file
  $self->_wantedTrack->{local_files} = [];

  my $pm = Parallel::ForkManager->new( $self->maxThreads );

  $pm->run_on_finish(
    sub {
      my ( $pid, $exitCode, $startId, $exitSig, $coreDump, $outFileRef ) = @_;

      if ( $exitCode != 0 ) {
        $self->log( 'fatal',
          "Failed to add dbnsfp, with exit code $exitCode for file $$outFileRef" );
      }

      push @{ $self->_wantedTrack->{local_files} }, path($$outFileRef)->basename;
    }
  );

  for my $file ( @{ $self->{_localFiles} } ) {
    $pm->start($file) and next;

    # Need to reset line endings here, or getReadFh may not operate correctly
    $self->setLineEndings("\n");

    my $fh = $self->getReadFh($file);
    my $outFh;

    $file =~ s/.gz$//;
    my $outFile = $file . '.with_dbnsfp.gz';

    $outFh = $self->getWriteFh($outFile);

    my $header = <$fh>;

    $self->setLineEndings($header);

    chomp $header;

    say $outFh join( "\t", $header, @dbNSFPheaderFields );

    while (<$fh>) {
      chomp;

      my @fields = split '\t', $_;

      my $foundDbNFSP;
      for my $field (@fields) {
        # Empirically determine
        if ( $dbNSFP{$field} ) {
          push @fields, @{ $dbNSFP{$field} };
          $foundDbNFSP = 1;
          last;
        }
      }

      if ( !$foundDbNFSP ) {
        push @fields, map { '.' } @dbNSFPheaderFields;
      }

      say $outFh join( "\t", @fields );
    }

    $pm->finish( 0, \$outFile );
  }

  $pm->wait_all_children();

  $self->_backupAndWriteConfig();
}

__PACKAGE__->meta->make_immutable;
1;
