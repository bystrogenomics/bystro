use 5.10.0;
use strict;
use warnings;

# The Bystro db contains only QC'd CADD sites
# Use these to filter the source CADD sites and output a cleaned version
package Utils::FilterCadd;

our $VERSION = '0.001';

use Mouse 2;
use namespace::autoclean;
use Types::Path::Tiny qw/AbsFile Path AbsDir/;
use Path::Tiny        qw/path/;
use Scalar::Util      qw/looks_like_number/;

use Seq::Tracks::Build::LocalFilesPaths;

use DDP;
use Parallel::ForkManager;

use Seq::DBManager;
use Seq::Tracks::Cadd;
use Seq::Tracks::Score::Build;

# Exports: _localFilesDir, _decodedConfig, compress, _wantedTrack, _setConfig, logPath, use_absolute_path
extends 'Utils::Base';

my $localFilesHandler = Seq::Tracks::Build::LocalFilesPaths->new();

sub BUILD {
  my $self = shift;

  # DBManager acts as a singleton. It is configured once, and then consumed repeatedly
  # However, in long running processes, this can lead to misconfiguration issues
  # and worse, environments created in one process, then copied during forking, to others
  # To combat this, every time Seq::Base is called, we re-set/initialzied the static
  # properties that create this behavior
  # Initialize it before BUILD, to make this class less dependent on inheritance order
  Seq::DBManager::initialize(
    { databaseDir => $self->_decodedConfig->{database_dir} } );

  if ( !$self->_wantedTrack->{sorted} == 1 ) {
    die "CADD files must be sorted (sorted == 1), at least by chromosome";
  }
}

sub go {
  my $self = shift;

  my $gzip = $self->gzip;

  my ( $localFilesPathsAref, $has_absolute_files ) =
    $localFilesHandler->makeAbsolutePaths(
    $self->_decodedConfig->{files_dir},
    $self->_wantedTrack->{name},
    $self->_wantedTrack->{local_files}
    );

  my $outDir =
    path( $self->_decodedConfig->{files_dir} )->child( $self->_wantedTrack->{name} );

  my $pm = Parallel::ForkManager->new( $self->maxThreads );

  if ( !@$localFilesPathsAref ) {
    $self->log( 'fatal', "No local files found" );
  }

  my %trackConfig = %{ $self->_wantedTrack };

  $trackConfig{assembly}    = $self->_decodedConfig->{assembly};
  $trackConfig{chromosomes} = $self->_decodedConfig->{chromosomes};

  # p %trackConfig;
  # exit;
  my $caddGetter = Seq::Tracks::Cadd->new( \%trackConfig );

  my $rounder = Seq::Tracks::Score::Build::Round->new(
    { scalingFactor => $caddGetter->scalingFactor } );

  my $db = Seq::DBManager->new();

  my %wantedChrs = map { $_ => 1 } @{ $self->_decodedConfig->{chromosomes} };

  my $dryRun = $self->dryRun;

  my $outExtPart = $self->compress ? '.txt.gz' : '.txt';
  my $outExt     = '.filtered' . $outExtPart;

  my @outPaths;

  # If we don't call the run_on_finish here
  # only 1 outPath will be stored for each fork, regardless of how many
  # files that fork read, since this will only be run once per termination
  # of a fork (rather than for each finish() call)
  $pm->run_on_finish(
    sub {
      my ( $pid, $exitCode, $liftedPath, $exitSignal, $coreDump, $outOrErrRef ) = @_;

      if ( $exitCode != 0 ) {
        $self->log( 'fatal',
          "$liftedPath failed liftOver due to: "
            . ( $outOrErrRef ? $$outOrErrRef : 'unknown error' ) );
        return;
      }

      $self->log( 'info', "PID $pid completed filtering $liftedPath to $$outOrErrRef \n" );
      push @outPaths, path($$outOrErrRef)->basename();
    }
  );

  for my $inPath (@$localFilesPathsAref) {
    my $outPath;

    my $outPathBase;
    # If this is a compressed file, strip the preceeding extension
    if ( $inPath =~ /.gz$/ ) {
      $outPathBase = substr( $inPath,      0, rindex( $inPath,      '.' ) );
      $outPathBase = substr( $outPathBase, 0, rindex( $outPathBase, '.' ) );
    }
    else {
      $outPathBase = substr( $inPath, 0, rindex( $inPath, '.' ) );
    }

    $pm->start($inPath) and next;
    my $readFh = $self->getReadFh($inPath);

    my $header = <$readFh>;
    $header .= <$readFh>;

    my $based    = 1;
    my $phredIdx = -1;
    my $altIdx   = -3;
    my $refIdx   = -4;

    if ( $header =~ 'chromStart' && $header =~ 'chromEnd' ) {
      $based = 0;
      say STDERR "$inPath is 0-based BED-Like";
    }

    my $outFh;

    my @fields;
    my $pos;

    my $dbVal;
    my $caddDbVals;
    my $caddDbScore = [];
    my $score;

    my $ref;
    my $alt;
    my $lastChr;
    my $chr;

    my $skipped = 0;

    # my $lastPosition;
    # my $scoreCount;

    # my $nonACTGrefCount;
    # my $nonACTGaltCount;
    # my $missingScore;
    while ( my $l = $readFh->getline() ) {
      chomp $l;
      @fields = split '\t', $l;

      #https://ideone.com/05wEAl
      #Faster than split
      #my $chr = substr($l, 0, index($l, "\t") );
      $chr = $fields[0];

      # May be unwanted if coming from CADD directly
      if ( !exists $wantedChrs{$chr} ) {
        #https://ideone.com/JDtX3z
        #CADD files don't use 'chr', but our cadd bed-like files do
        if ( index( $chr, 'chr' ) == -1 ) {
          $chr = 'chr' . $chr;
        }

        #Our bed-like file will use 'chrM', but the original cadd file will have 'chrMT'
        if ( $chr eq 'chrMT' ) {
          $chr = 'chrM';
        }

        # Check again that this is unwanted
        if ( !exists $wantedChrs{$chr} ) {
          $self->log( 'warn', "Skipping unwanted: $chr" );

          $skipped++;
          next;
        }
      }

      if ( defined $lastChr ) {
        if ( $lastChr ne $chr ) {
          $db->cleanUp();

          my $err = "Expected only a single chromosome, found $chr and $lastChr";
          $self->log( 'error', $err );

          $pm->finish( 255, \$err );
        }
      }
      else {
        $lastChr = $chr;
      }

      my $pos = $fields[1];

      if ( !defined $pos ) {
        $db->cleanUp();

        my $err = 'Undefined position';
        $self->log( 'error', $err );

        $pm->finish( 255, \$err );
      }

      if ( !$outFh && !$dryRun ) {
        $outPath = "$outPathBase.$chr$outExt";

        $self->log( 'info', "Found $chr in $inPath; creating $outPath" );

        if ( -e $outPath && !$self->overwrite ) {
          $self->log( 'warn',
            "outPath $outPath exists, skipping $inPath because overwrite not set" );
          last;
        }

        $outFh = $self->getWriteFh($outPath);

        print $outFh $header;
      }

      $dbVal = $db->dbReadOne( $chr, $pos - $based );

      # We expect the db to have a value, but it's possible the CADD track gave us a nonsense pos
      # This is unusual, so log
      if ( !defined $dbVal ) {
        $self->log( 'warn', "Couldn't find a value for $chr\:$pos ($based\-based)" );

        $skipped++;
        next;
      }

      $caddDbVals = $dbVal->[ $caddGetter->dbName ];

      # A value that failed QC, and wasn't included in the db
      # Not unusual, don't log
      if ( !defined $caddDbVals ) {
        $skipped++;
        next;
      }

      if ( defined $caddDbVals && @$caddDbVals != 3 ) {
        $db->cleanUp();

        my $err =
            "Couldn't find 3 cadd values for $chr\:$pos ($based\-based) ... Found "
          . ( scalar @$caddDbVals )
          . " instead";
        $self->log( 'error', \$err );

        $pm->finish( 255, \$err );
      }

      # Everything with $caddDbVals should be a well-qc'd base
      # If not, die
      $ref = $fields[$refIdx];
      $alt = $fields[$altIdx];

      if ( $ref ne 'A' && $ref ne 'C' && $ref ne 'T' && $ref ne 'G' ) {
        $db->cleanUp();

        my $err = "$chr\:$pos ($based\-based) : Expected ACTG ref, found $ref";
        $self->log( 'error', \$err );

        $pm->finish( 255, \$err );
      }

      if ( $alt ne 'A' && $alt ne 'C' && $alt ne 'T' && $alt ne 'G' ) {
        $db->cleanUp();

        my $err = "$chr\:$pos ($based\-based) : Expected ACTG alt, found $alt";
        $self->log( 'error', \$err );

        $pm->finish( 255, \$err );
      }

      $score = $fields[$phredIdx];

      if ( !looks_like_number($score) ) {
        $db->cleanUp();

        my $err = "$chr\:$pos ($based\-based) : Expected numerical PHRED, found $score";
        $self->log( 'error', \$err );

        $pm->finish( 255, \$err );
      }

      $caddDbScore = $caddGetter->get( $dbVal, $chr, $ref, $alt, 0, $caddDbScore );

      # We round the score to check against the db-held value, which is rounded
      if ( $rounder->round($score) != $rounder->round( $caddDbScore->[0] ) ) {
        $db->cleanUp();

        my $err = "$chr\:$pos ($based\-based) : Expected PHRED $caddDbScore->[0], found: "
          . $rounder->round($score);
        $self->log( 'error', \$err );

        $pm->finish( 255, \$err );
      }

      if ( !$dryRun ) {
        say $outFh $l;
      }
    }

    $self->log( 'info', "Skipped $skipped sites in $inPath" );
    $pm->finish( 0, \$outPath );
  }

  $pm->wait_all_children();

  $self->_wantedTrack->{local_files} = \@outPaths;

  $self->_backupAndWriteConfig();

  return 1;
}

__PACKAGE__->meta->make_immutable;
1;
