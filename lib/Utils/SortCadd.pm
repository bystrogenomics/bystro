use 5.10.0;
use strict;
use warnings;

use lib '../';
# Takes a yaml file that defines one local file, and splits it on chromosome
# Only works for tab-delimitd files that have the c
package Utils::SortCadd;

our $VERSION = '0.001';

use Mouse 2;
use namespace::autoclean;
use Path::Tiny qw/path/;

use DDP;

use Seq::Tracks::Build::LocalFilesPaths;
use Parallel::ForkManager;

# _localFilesDir, _decodedConfig, compress, _wantedTrack, _setConfig, and logPath, 
extends 'Utils::Base';

########## Arguments accepted ##############
# Takes either a bed-like cadd file, or a cadd file, and sorts it
# Works a bit differently from CaddToBed or LiftOverCadd, in that 
# even if the input file is compressed, by default we compress the output
# Otherwise the intermediate files generated before sort become enormous, and this run could take over 600GB
has '+compress' => (default => 1);

my $localFilesHandler = Seq::Tracks::Build::LocalFilesPaths->new();

sub BUILD {
  my $self = shift;

  $self->_wantedTrack->{local_files} = $localFilesHandler->makeAbsolutePaths($self->_decodedConfig->{files_dir},
    $self->_wantedTrack->{name}, $self->_wantedTrack->{local_files});
}

sub sort {
  my $self = shift;

  my %wantedChrs = map { $_ => 1 } @{ $self->_decodedConfig->{chromosomes} };
    
  # record out paths so that we can unix sort those files
  my @outPaths;
  my %outFhs;

  $self->log('info', "Beginning organizing cadd files by chr (single threaded)");

  my $outExtPart = $self->compress ? '.txt.gz' : '.txt';

  my $outExt = '.organized-by-chr' . $outExtPart;

  for my $inFilePath ( @{$self->_wantedTrack->{local_files} } ) {
    my $outPathBase = substr($inFilePath, 0, rindex($inFilePath, '.') );

    $outPathBase =~ s/\.(chr[\w_\-]+)//;
    
    # Store output handles by chromosome, so we can write even if input file
    # out of order
    $self->log('info', "Out path base: $outPathBase");
    $self->log('info', "Reading input file: $inFilePath");
    
    my (undef, $compressed, $readFh) = $self->get_read_fh($inFilePath);

    my $versionLine = <$readFh>;
    my $headerLine = <$readFh>;

    $self->log('info', "Read version line: $versionLine");
    $self->log('info', "Read header line: $headerLine");

    while(my $l = $readFh->getline() ) {
      #https://ideone.com/05wEAl
      #Faster than split
      my $chr = substr($l, 0, index($l, "\t") );

      # May be unwanted if coming from CADD directly
      if(!exists $wantedChrs{$chr}) {
        #https://ideone.com/JDtX3z
        #CADD files don't use 'chr', but our cadd bed-like files do
        if (substr($chr, 0, 3) ne 'chr') {
          $chr = 'chr' . $chr;
        }

        #Our bed-like file will use 'chrM', but the original cadd file will have 'chrMT'
        if($chr eq 'chrMT') {
          $chr = 'chrM';
        }

        # Check again that this is unwanted
        if(!exists $wantedChrs{$chr}) {
          $self->log('warn', "Skipping unwanted: $chr");
          next;
        }
      }

      my $fh = $outFhs{$chr};

      if(!$fh) {
        my $outPath = "$outPathBase.$chr$outExt";

        $self->log('info', "Found $chr in $inFilePath; creating $outPath");
        
        push @outPaths, $outPath;

        if(-e $outPath && !$self->overwrite) {
          $self->log('warn', "outPath $outPath exists, skipping $inFilePath because overwrite not set");
          last;
        }

        $outFhs{$chr} = $self->get_write_fh($outPath);

        $fh = $outFhs{$chr};

        print $fh $versionLine;
        print $fh $headerLine;
      }
      
      print $fh $l;
    }
  }
  
  for my $outFh (values %outFhs) {
    close $outFh;
  }

  $self->log('info', "Finished organizing cadd files by chr, beginning sort (multi threaded)");

  # TODO: use max processors based on # of cores
  my $pm = Parallel::ForkManager->new(8);

  my @finalOutPaths;

  for my $outPath (@outPaths) {
    my $gzipPath = $self->gzip;

    my (undef, $compressed, $fh) = $self->get_read_fh($outPath);

    my $outExt = '.sorted' . $outExtPart;

    my $finalOutPathBase = substr($outPath, 0, rindex($outPath, '.') );

    my $finalOutPath = $finalOutPathBase . $outExt;

    my $tempPath = path($finalOutPath)->parent()->stringify;

    $pm->start($finalOutPath) and next;
      my $command;

      #k2,2 means sort only by column 2. column 2 is either chrStart or Pos
      if($compressed) {
        $command = "( head -n 2 <($gzipPath -d -c $outPath) && tail -n +3 <($gzipPath -d -c $outPath) | sort --compress-program $gzipPath -T $tempPath -k2,2 -n ) | $gzipPath -c > $finalOutPath";
      } else {
        $command = "( head -n 2 $outPath && tail -n +3 $outPath | sort --compress-program $gzipPath -T $tempPath -k2,2 -n ) > $finalOutPath";
      }

      $self->log('info', "Running command: $command");

      my $exitStatus = system(("bash", "-c", $command));

      if($exitStatus == 0) {
        $self->log('info', "Successfully finished sorting $outPath. Exit status: $exitStatus");
        $exitStatus = system("rm $outPath");
      } else {
        $self->log('error', "Failed to sort $outPath. Exit status: $exitStatus. Expect fatal message and program exit.");
      }
    # returns the exit status for run_on_finish to die
    $pm->finish($exitStatus);
  }

  $pm->run_on_finish(sub {
    my ($pid, $exitCode, $finalOutPath) = @_;

    if($exitCode != 0) {
      return $self->log('fatal', "$finalOutPath failed to sort, with exit code $exitCode");
    }
    push @finalOutPaths, path($finalOutPath)->basename;
  });

  $pm->wait_all_children();

  $self->_wantedTrack->{local_files} = \@finalOutPaths;

  $self->_wantedTrack->{sortCadd_date} = $self->_dateOfRun;

  # Make sure that we indicate to the user that cadd is guaranteed to be sorted
  # This speeds up cadd building
  $self->_wantedTrack->{sorted_guaranteed} = 1;

  $self->_backupAndWriteConfig();
}

__PACKAGE__->meta->make_immutable;
1;
