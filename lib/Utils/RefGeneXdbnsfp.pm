use 5.10.0;
use strict;
use warnings;

use lib '../';

# Adds dbnsfp to refGene
package Utils::RefGeneXdbnsfp;

our $VERSION = '0.001';

use Mouse 2;
use namespace::autoclean;
use Path::Tiny qw/path/;
use Parallel::ForkManager;

use Seq::Tracks::Build::LocalFilesPaths;
use DDP;

# # _localFilesDir, _decodedConfig, compress, _wantedTrack, _setConfig, and logPath
extends 'Utils::Base';

# ########## Arguments accepted ##############
# Expects tab delimited file; not allowing to be set because it probably won't ever be anything
# other than tab, and because split('\'t') is faster
# has delimiter => (is => 'ro', lazy => 1, default => "\t");
has geneFile => (is => 'ro', isa => 'Str', required => 1);

sub BUILD {
  my $self = shift;

  my $localFilesHandler = Seq::Tracks::Build::LocalFilesPaths->new();

   $self->{_localFiles} = $localFilesHandler->makeAbsolutePaths($self->_decodedConfig->{files_dir},
    $self->_wantedTrack->{name}, $self->_wantedTrack->{local_files});

   if(!@{$self->{_localFiles}}) {
    $self->log('fatal', "Require some local files");
   }
}

# TODO: error check opening of file handles, write tests
sub go {
  my $self = shift;

  # Store output handles by chromosome, so we can write even if input file
  # out of order
  my %outFhs;
  my %skippedBecauseExists;

  my $dbnsfpFh = $self->get_read_fh($self->geneFile);

  my $header = <$dbnsfpFh>;
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
      if($dCol eq $col) {
        push @geneNameIdx, $idx;
      }

      $idx++;
    }
  }

  # namespace
  @dbNSFPheaderFields = map { 'dbnsfp.' . $_ } @dbNSFPheaderFields;

  my %dbNSFP;
  while(<$dbnsfpFh>) {
    #super chomp; also helps us avoid weird characters in the fasta data string
    #helps us find shitty lines
    $_ =~ s/\s+$//;
    my @fields = split '\t', $_;
    if(@fields != @dbNSFPheaderFields) {
      $self->log('fatal', "WTF: $_");
    }

    my $i = -1;
    for my $idx (@geneNameIdx) {
      $i++;

      my @vals = split ';', $fields[$idx];

      # sometimes dbNSFP gives duplicate values in the same string...
      my %seenThisLoop;
      for my $val (@vals) {
        if($val eq '.' || $val !~ /^\w+/) {
          $self->log('fatal', "WTF: missing gene?");
        }

        $seenThisLoop{$val} = 1;

        if(exists $dbNSFP{$val}) {
          $self->log('fatal', "Duplicate entry found: $val, skipping : $_");
          next;
        }

        $dbNSFP{$val} = \@fields;
      }
    }
  }

  # We'll update this list of files in the config file
  $self->_wantedTrack->{local_files} = [];

  my $pm = Parallel::ForkManager->new($self->maxThreads);

  $pm->run_on_finish(sub {
    my ($pid, $exitCode, $startId, $exitSig, $coreDump, $outFileRef) = @_;

    if($exitCode != 0) {
      $self->log('fatal', "Failed to add dbnsfp, with exit code $exitCode for file $$outFileRef");
    }

    push @{$self->_wantedTrack->{local_files}}, $$outFileRef;
  });

  for my $file (@{$self->{_localFiles}}) {
    $pm->start($file) and next;

    my $fh = $self->get_read_fh($file);
    my $outFh; 

    $file =~ s/.gz$//;
    my $outFile = $file . '.with_dbnsfp.gz';

    $outFh = $self->get_write_fh($outFile);

    my $header = <$fh>;
    chomp $header;

    say $outFh join("\t", $header, @dbNSFPheaderFields);

    while(<$fh>) {
      chomp;

      my @fields = split '\t', $_;

      my $foundDbNFSP;
      for my $field (@fields) {
        # Empirically determine
        if($dbNSFP{$field}) {
          push @fields, @{$dbNSFP{$field}};
          $foundDbNFSP = 1;
          last;
        }
      }

      if(!$foundDbNFSP) {
        push @fields, map { '.' } @dbNSFPheaderFields;
      }

      say $outFh join("\t", @fields);
    }

    $pm->finish(0, \$outFile);
  }

  $pm->wait_all_children();

  # TODO: store completion under the util object
  $self->_wantedTrack->{refGeneXdbnsfp_date} = $self->_dateOfRun;

  $self->_backupAndWriteConfig();
}

__PACKAGE__->meta->make_immutable;
1;
