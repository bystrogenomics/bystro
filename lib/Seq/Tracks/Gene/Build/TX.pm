use 5.10.0;
use strict;
use warnings;

# ABSTRACT: Creates a hash reference of sites with information in them
=head1 DESCRIPTION
  # A consuming build class stores this info at some key in the main database
=cut

# Inspired by Thomas Wingo's similar in-silico transcrpion Seqant 3 package

#TODO: Thomas. I may have misunderstood your intentions here. I spoke with
#Dave, and we decided, I believe, not to write peptides, because those 
#were already reconstructed at annotation time (codon2AA)
#If I made significant mistakes here, 1000 apologies. I did my best
#to follow your code base, and the changes made were done for a few reasons:
#1) Make it easier for me to understand
#2) Make it clearer to future developers exactly what this class is responsible for
#3) Reduce the amount of code to the minimum needed to output a transcriptSite
#4) Related to #3, don't use moose attributes unless need $self or exposed as public API

package Seq::Tracks::Gene::Build::TX;

our $VERSION = '0.001';

use Mouse 2;
use MouseX::NativeTraits;
# We need pre-initialized tracks
use Seq::Tracks;
use Seq::Tracks::Gene::Site;
use Seq::DBManager;

with 'Seq::Role::Message';

use namespace::autoclean;
use DDP;

#how many bases away from exon bound we will call spliceAc or spliceDon site
my $spliceSiteLength = 2;
#placeholder for annotation in string
my $annBase = '0';

# stores all of our individual sites
# these can be used by the consumer to write per-reference-position
# codon information
# The only public variable other than transcriptErrors, which we may discard
has transcriptSites => (
  is      => 'ro',
  isa     => 'ArrayRef',
  lazy => 1,
  default => sub { [] },
  init_arg => undef,
);

# Also public (for now)
# uses transcriptAnnotations to figure out if anything went wrong
has transcriptErrors => (
  is      => 'rw',
  isa     => 'ArrayRef',
  writer => '_writeTranscriptErrors',
  traits  => ['Array'],
  handles => {
    noTranscriptErrors   => 'is_empty',
    allTranscriptErrors => 'elements',
  },
  lazy    => 1,
  default => sub { [] },
  init_arg => undef,
);

###All required arguments
has chrom => (
  is => 'ro',
  isa => 'Str',
  required => 1,
);

has exonStarts => (
  is => 'ro',
  isa => 'ArrayRef',
  traits => ['Array'],
  handles => {
    allExonStarts => 'elements',
  },
  required => 1,
);

has exonEnds => (
  is => 'ro',
  isa => 'ArrayRef',
  traits => ['Array'],
  handles => {
    allExonEnds=> 'elements',
  },
  required => 1,
);

has cdsStart => (
  is => 'ro',
  isa => 'Str',
  required => 1,
);

has cdsEnd => (
  is => 'ro',
  isa => 'Str',
  required => 1,
);

has strand => (
  is => 'ro',
  isa => 'Str',
  required => 1,
);

has txNumber => (
  is => 'ro',
  isa => 'Int',
  required => 1,
);

##End required arguments
#purely for debug
#not the same as the Track name
has name => (
  is => 'ro',
  isa => 'Str',
  required => 1,
);

##End requ
###private
has debug => (
  is => 'ro',
  isa => 'Int',
  default => 0,
  lazy => 1,
);

#@private
state $codonPacker = Seq::Tracks::Gene::Site->new();

#coerce our exon starts and ends into an array
sub BUILDARGS {
  my ($orig, $href) = @_;

  # The original data is a comma-delimited string
  # But since Bystro often coerces delimited things into arrays,
  # We may be given an array instead; If not, coerce into an array
  if(!ref $href->{exonStarts}) {
    $href->{exonStarts} = [ split(',', $href->{exonStarts} ) ];
  }

  if(!ref $href->{exonEnds}) {
    $href->{exonEnds} = [ split(',', $href->{exonEnds} ) ];
  }

  return $href;
};

sub BUILD {
  my $self = shift;

  $self->log('debug', "Beginning to build tx: " . $self->name);

  #seeds transcriptSequence and transcriptPositions
  my ($seq, $seqPosMapAref) = $self->_buildTranscript();

  my $txAnnotationHref = $self->_buildTranscriptAnnotation();

  # say "txAnnotationHref";
  # p $txAnnotationHref;
  my $errorsAref = $self->_buildTranscriptErrors($seq, $seqPosMapAref, $txAnnotationHref);
  #if errors warn; some transcripts will be malformed
  #we could pass an array reference to log, but let's give some additional 
  #context
  if(@$errorsAref) {
    my $error = 'For tx ' . $self->name. ' : ' . join('; ', @$errorsAref);
    $self->log('warn', $error);  
  }

  #We no longer strictly need to set 'abs_pos' for each exon_end
  #We *could* subtract $based from each position, but skipping for now because
  #it seems unlikely that we'll start accepting 1-based gene tracks
  #maybe if UCSC loses it's dominance

  # transcriptSites holds all of our site annotations
  $self->_buildTranscriptSites($seq, $seqPosMapAref, $txAnnotationHref);

  #this is now handled as part of the buildTranscript process
  #$self->_build_flanking_sites;
}

# give the sequence with respect to the direction of transcription / coding
# at the same time, creates an array of the positions to which that transcript
# belongs, and assigns what we call that thing

#this combines _build_transcript_db, _build_flanking_sites, _build_transcript_abs_position
#since those mostly do the exact same thing, except in what they store
#and flanking sites is really adding data to the transcript_annotation

#note that we no longer have absolute positions, so we don't call them absolute
#(because we're storing position by chr in the kv database)
sub _buildTranscript {
  my $self        = shift;
  my @exonStarts = $self->allExonStarts;
  my @exonEnds   = $self->allExonEnds;
  my $codingStart = $self->cdsStart;
  my $codingEnd = $self->cdsEnd;

  my (@sequencePositions, $txSequence);

  my $tracks = Seq::Tracks->new();

  my $refTrack = $tracks->getRefTrackGetter();

  # Expects DBManager to have been previously configured
  my $db = Seq::DBManager->new();

  #in scalar, as in less than, @array gives length
  for ( my $i = 0; $i < @exonStarts; $i++ ) {
    if ( $exonStarts[$i] >= $exonEnds[$i] ) {
      $self->log('fatal', "exon start $exonStarts[$i] >= end $exonEnds[$i]");
    }

    #exonEnds is closed, so the actual exonEnd is - 1
    #exonStarts are open
    #http://genomewiki.ucsc.edu/index.php/Coordinate_Transforms

    #transcript starts are 0-based and ends are 1-based
    #http://genome.ucsc.edu/FAQ/FAQtracks#tracks1
    #perl range is always closed
    #https://ideone.com/AKKpfC
    # my @fragmentPositions;

    my $exonPosHref = [ $exonStarts[$i] .. $exonEnds[$i] - 1 ];

    # say "exon start was " . $exonStarts[$i];
    # say "exon end was " . ($exonEnds[$i] - 1);
    # say "from that we got exonPosHref";
    # p $exonPosHref;
    #limitation of the below API; we need to copy $posHref
    #thankfully, we needed to do this anyway.
    #we push them in order, so the first position in the 
    #https://ideone.com/wq0YJO (dereferencing not strictly necessary, but I think clearer)
    push @sequencePositions, @$exonPosHref;

    # say "sequence positions are";
    # p @sequencePositions;

    # As a result of modifying the reference, each position in exonPosHref
    # now has database data, or undefined
    # last argument means don't commit this, saves io overhead,
    # committing here to avoid interfering with transactions of the consuming class
    # TODO: allow dbRead to use read-only transactions
    $db->dbRead($self->chrom, $exonPosHref);

    #Now get the base for each item found in $dAref ($exonPosHref);
    #This is handled by the refTrack of course
    #Each track has its own "get" method, which fetches its data
    #That can be a scalar or a hashRef
    #Ref tracks always return a scalar, a single base, since that's the only
    #thing that they could return

    #This doesn't work for some reason.
    #https://ideone.com/1sJC69
    #$txSequence .= reduce { ref $a ? $refTrack->get($a) : $a . $refTrack->get($b) } @$dAref (@$exonPosHref)
    # say "length is " . scalar @$dAref (@$exonPosHref);
    # exit;
    for (my $y = 0; $y < scalar @$exonPosHref; $y++) {
      my $refBase = $refTrack->get( $exonPosHref->[$y] );

      if(!$refBase) {
        my @positionsWeHad = ($exonStarts[$i] .. $exonEnds[$i] - 1);
        $self->log('fatal', "Position $positionsWeHad[$y] doesn't exist in our "
          . $self->chrom . " database."
         . "\nWe've either selected the wrong assembly,"
         . "\nor haven't built the reference database for this chromosome" );
      }

      $txSequence .= $refBase;
    }

    # The above replaces this from _build_transcript_db; 
    # note that we still respect half-closed exonEnds range
    # for ( my $i = 0; $i < @exon_starts; $i++ ) { #half closed
    #   my $exon;
    #   for ( my $abs_pos = $exon_starts[$i]; $abs_pos < $exon_ends[$i]; $abs_pos++ ) {
    #     $exon .= $self->get_base( $abs_pos, 1 );
    #   }
    #   # say join ("\t", $exon_starts[$i], $exon_ends[$i], $exon);
    #   $seq .= $exon;
    # }
  }

  if ( $self->strand eq "-" ) {
    #reverse the sequence, just as in _build_transcript_db
    $txSequence = reverse $txSequence;
    # get the complement, just as in _build_transcript_db
    $txSequence =~ tr/ACGT/TGCA/;
    #reverse the positions, just as done in _build_transcript_abs_position
    @sequencePositions = reverse @sequencePositions;
  } 

  return ($txSequence, \@sequencePositions); 

  #now in buildTranscriptAnnotation
  #my $errorsAref = $self->_buildTranscriptErrors($txSequence, \@sequencePositions);
}

sub _buildTranscriptAnnotation {
  my $self = shift;

  my @exonStarts = $self->allExonStarts;
  my @exonEnds   = $self->allExonEnds;
  my $codingStart = $self->cdsStart;
  my $codingEnd = $self->cdsEnd;

  my $posStrand = $self->strand eq '+';
  #https://ideone.com/B3ygW6
  #is this a bug? isn't cdsEnd open, so shouldn't it be cdsStart == cdsEnd - 1
  #nope: http://genome.soe.ucsc.narkive.com/NHHMnfwF/cdsstart-cdsend-definition
  my $nonCoding = $self->cdsStart == $self->cdsEnd;

  my $txAnnotationHref;

  # Store intron site type if that's what this is
  # Note that the splice loop below this will ovewrite any positions that are
  # Called Splice Sites
  INTRON_LOOP: for ( my $i = 0; $i < @exonEnds; $i++ ) {
    my $thisExonEnd = $exonEnds[$i];
    my $nextExonStart = $exonStarts[$i + 1];

    if(!$nextExonStart) {
      last INTRON_LOOP;
    }
    #exon Ends are open, so the exon actually ends $exonEnds - 1
    for (my $intronPos = $thisExonEnd; $intronPos < $nextExonStart; $intronPos++ ) {
      $txAnnotationHref->{$intronPos} = $codonPacker->siteTypeMap->intronicSiteType;
    }
  }

  #Then store non-coding, 5'UTR, 3'UTR annotations
  for (my $i = 0; $i < @exonStarts; $i++) {
    # Annotate splice donor/acceptor bp
    #  - i.e., bp within $spliceSiteLength bp of exon start / stop
    #  - what we want to capture is the bp that are within $spliceSiteLength bp of the start or end of
    #    an exon start/stop; whether this is only within the bounds of coding exons does
    #    not particularly matter to me
    #
    # From the gDNA:
    #
    #        EStart    CStart          EEnd       EStart    EEnd      EStart   CEnd      EEnd
    #        +-----------+---------------+-----------+--------+---------+--------+---------+
    #  Exons  111111111111111111111111111             22222222           333333333333333333
    #  Code               *******************************************************
    #  APR                                        ###                ###
    #  DNR                                %%%                  %%%
    #

    #Compared to the "Seq" codebase written completely by Dr. Wingo: 
    #The only change to the logic that I have made, is to add a bit of logic
    #to check for cases when our splice donor / acceptor sites 
    #as calculated by a the use of $spliceSiteLength
    #actually overlap either the previous exonEnd, or the next exonStart
    #and are therefore inside of a coding sequence.
    #Dr. Wingo's original solution was completely correct, because it also 
    #assumed that downstream someone was smart enough to intersect 
    #coding sequences and splice site annotations, and keep the coding sequence
    #in any overlap

    #TODO: should we check if start + n is past end? or >= end - $n

    #If $i == 0, make sure we don't accidentally call the last exonEnd the first one
    my $previousExonEnd = $i > 0 ? $exonEnds[$i-1] : undef;
    my $nextExonStart = $exonStarts[$i+1];

    # We cannot have a spliceAcceptor site unless there was an upstream exon
    if(defined $previousExonEnd) {
      # And if the length of the intron is smaller than 2*spliceSiteLength
      # we would have overlap between our spliceAcceptor and donor, which doesn't make sense
      #exonEnd is open, so previousExonEnd is the first base of the intron
      #and #exonStart is closed, so it is +1 the end of the intron
      #Also, exonStarts, ends always relative to sense strand
      #So ends always larger than starts
      if($exonStarts[$i] - $previousExonEnd >= 2 * $spliceSiteLength) {
        for ( my $n = 1; $n <= $spliceSiteLength; $n++ ) {

          # The first base of the spliceAcceptor is up to $spliceSiteLength bases
          # away from the downstream exonStart
          # exonStart is closed, meaning the exonStart is the first base of the exon
          my $exonPos = $exonStarts[$i] - $n;

          #This last condition to prevent splice acceptors for being called in
          #coding sites for weirdly tight transcripts
          # >= because EEnd (exonEnds) are open range, aka their actual number is not
          #to be included, it's 1 past the last base of that exon
          #(and exonEnds are open, so exonEnd is the first base of the downstream intron)
          $txAnnotationHref->{$exonPos} =
              $posStrand
              ? $codonPacker->siteTypeMap->spliceAcSiteType
              : $codonPacker->siteTypeMap->spliceDonSiteType;
        }
      } else {
        $self->log('warn', "No spliceAcceptor (neg donor) possible: intron between exons $i and ". ($i - 1) . " of " . $self->name . " is only ". ($exonStarts[$i] - $previousExonEnd) . " bp long");
      }
    }

    if (defined $nextExonStart) {
      #exonEnd is the first intron, exonStart is +1 of the intron
      if( $nextExonStart - $exonEnds[$i] >= 2 * $spliceSiteLength) {
        for ( my $n = 1; $n <= $spliceSiteLength; $n++ ) {
          # The exonEnd is already the first intron, which means the first spliceDonor site
          # so we subtract 1 and add however many bases of spliceSiteLength wanted
          my $exonPos = $exonEnds[$i] - 1 + $n;

          $txAnnotationHref->{$exonPos} =
            $posStrand
            ? $codonPacker->siteTypeMap->spliceDonSiteType
            : $codonPacker->siteTypeMap->spliceAcSiteType;
        }
      } else {
        $self->log('warn', "No spliceDonor (neg acceptor) possible: intron between exons $i and ". ($i + 1) . " of " . $self->name . " is only ". ($nextExonStart - $exonEnds[$i]) . " bp long");
      }
    }

    #I give UTR regions the lowest precedence at the moment;
    #Meaning if the site is already marked spliceDonor/Acceptor or intronic that won't
    #be overriden
    #In the future, we could define UTR5-intronic, UTR5-spliceDonor, etc
    #To do this we would just iterate pos between cdsStart and cdsEnd
    UTR_NCRNA_LOOP: for ( my $exonPos = $exonStarts[$i]; $exonPos < $exonEnds[$i]; $exonPos++ ) {
      if($nonCoding) {
        $txAnnotationHref->{$exonPos} = $codonPacker->siteTypeMap->ncRNAsiteType;

        next UTR_NCRNA_LOOP;
      }

      # Don't overwrite intronic or spliceDonor/Acceptor
      if($txAnnotationHref->{$exonPos}) {
        next UTR_NCRNA_LOOP;
      }

      #Next calculate UTR regions. We call utr regions from cdsStart to cdsEnd (cdsEnd is oddly closed)
      #TODO this may be a subtle bug, I think it shuold be $pos <= $codingEnd
      #On second thought, it looks fine. codingEnd is treated as closed here
      if( $exonPos < $codingEnd ) {
        if( $exonPos >= $codingStart ) {
          #It's in the body of the translated region
          next UTR_NCRNA_LOOP;
        }

        $txAnnotationHref->{$exonPos} =
          $posStrand
          ? $codonPacker->siteTypeMap->fivePrimeSiteType
          : $codonPacker->siteTypeMap->threePrimeSiteType;
      } else {
        #if we're after cds end, but in an exon we must be in the 3' UTR
        $txAnnotationHref->{$exonPos} =
        $posStrand
        ? $codonPacker->siteTypeMap->threePrimeSiteType
        : $codonPacker->siteTypeMap->fivePrimeSiteType;
      }
    }
  }

  #my $errorsAref = $self->_buildTranscriptErrors($txSequence, $txAnnotationHref);

  return $txAnnotationHref;
  #return ($txAnnotationHref, $errorsAref);
}

#TODO: double check that this works properly
#TODO: think about whether we want to pack anything if no info
#the problem with not packing something is we won't know how to unpack it apriori
sub _buildTranscriptSites {
  my ($self, $txSequence, $seqPosMapAref, $txAnnotationHref) = @_;
  my @exonStarts       = $self->allExonStarts;
  my @exonEnds         = $self->allExonEnds;

  # we build up our site annotations in 2 steps
  # 1st record everything as if it were a coding sequence
  # then we overwrite those entries if there were other annotations associated
  # advantage is we can keep 3-mer info for those annotated sites
  # and don't need to modify the $txSequence string in _buildTranscrtipAnnotation
  # which may be (for me) a bit easier to reason about
  my %tempTXsites;

  #First add the annotations; note that if for some reason a codon overlaps
  for my $pos (keys %$txAnnotationHref) {
    my $siteType =  $txAnnotationHref->{$pos};

    #storing strand for now, could remove it later if we decided to 
    #just get it from the region database entry for the transcript
    $tempTXsites{$pos} = [$self->txNumber, $siteType, $self->strand, undef, undef, undef];
  }

  my $codingBaseCount = 0;
  #Then, make all of the codons in locations that aren't in the $tempTXsites

  #Example (informal test): #https://ideone.com/a9NYhb
  CODING_LOOP: for (my $i = 0; $i < length($txSequence); $i++ ) {
    #get the genomic position
    my $pos = $seqPosMapAref->[$i];

    if(defined $tempTXsites{$pos} ) {
      next CODING_LOOP;
    }

    my ($siteType, $codonNumber, $codonPosition, $codonSeq);

    # Next check any remaining sites. These should be coding sites.
    # We check the base composition (ATCG) to make sure no oddiities slipped by
    # At this point, any remaining sites should be in the coding region
    # Since we've accounted for non-coding, UTR, and ~ splice sites

    if ( substr($txSequence, $i, 1) =~ m/[ACGT]/ ) {
      #the codon number ; POSIX::floor safer than casting int for rounding
      #but we just want to truncate; http://perldoc.perl.org/functions/int.html
      $codonNumber   = 1 + int( $codingBaseCount / 3 );

      $codonPosition = $codingBaseCount % 3;

      my $codonStart = $i - $codonPosition;

      # Replaces: #for ( my $j = $codonStart; $j <= $codonEnd; $j++ ) {  #$referenceCodonSeq .= $self->getTranscriptBases( $j, 1 ); #}
      $codonSeq = substr( $txSequence, $codonStart, 3 );

      $siteType = $codonPacker->siteTypeMap->codingSiteType;

      $tempTXsites{ $pos } = [$self->txNumber, $siteType, $self->strand,
       $codonNumber, $codonPosition, $codonSeq];

      $codingBaseCount++;
      next CODING_LOOP;
    }

    $self->log('warn', substr($txSequence, $i, 1) . " at $pos in transcript "
      . $self->name . " not A,T,C, or G");
  }

  #At this point, we have all of the codon information stored.
  #However, some sites won't be coding, and those are in our annotation href

  #Now compact the site details
  for my $pos (sort {$a <=> $b} keys %tempTXsites) {
    #stores the codon information as binary
    #this was "$self->add_transcript_site($site)"
    # passing args in list context 
    # https://ideone.com/By1GDW
    my $site = $codonPacker->pack( @{$tempTXsites{$pos} } );

    #this transcript sites are keyed on reference position
    #this is similar to what was done with Seq::Site::Gene before
    push @{$self->transcriptSites}, $pos, $site;
  }
}

# check coding sequence is
#   1. divisible by 3
#   2. starts with ATG
#   3. Ends with stop codon
sub _buildTranscriptErrors {
  my $self = shift;
  my $seq =  shift;
  my $seqPosAref = shift;
  my $transcriptAnnotationHref = shift;

  state $atgRe = qr/\AATG/; #starts with ATG
  state $stopCodonRe = qr/(TAA|TAG|TGA)\Z/; #ends with a stop codon

  my @errors = ();

  if ( $self->cdsStart == $self->cdsEnd ) {
    #it's a non-coding site, so it has no sequence information stored at all
    return \@errors;
  }

  #I now see why Thomas replaced bases in the exon seq with 5 and 3
  my $codingSeq;
  for(my $i = 0; $i < length($seq); $i++) {
    if(defined $transcriptAnnotationHref->{ $seqPosAref->[$i] } ) {
      next;
    }
    $codingSeq .= substr($seq, $i, 1);
  }

  my $codingSeq2;
  for(my $i = 0; $i < length($seq); $i++) {
    if($seqPosAref->[$i] >= $self->cdsStart && $seqPosAref->[$i] < $self->cdsEnd) {
      $codingSeq2 .= substr($seq, $i, 1);
    }
  }

  if($codingSeq ne $codingSeq2) {
    if($self->debug) {
      say "condingSeq ne codingSeq2";
      say "coding seq is: ";
      p $codingSeq;
      say "coding seq length: ";
      my $length = length($codingSeq);
      p $length;
      say "coding seq 2 is: ";
      p $codingSeq2;
      say "coding seq 2 length: ";
      $length = length($codingSeq2);
      p $length;
      say "name of transcript:";
      p $self->name;
      say "strand is";
      p $self->strand;

      my $numSpliceStuff = 0;
      for my $pos (keys %$transcriptAnnotationHref) {
        if($transcriptAnnotationHref->{$pos} eq $codonPacker->siteTypeMap->spliceAcSiteType || 
          $transcriptAnnotationHref->{$pos} eq $codonPacker->siteTypeMap->spliceDonSiteType) {
          $numSpliceStuff++;
        }
      }

      say "difference in length: " . (length($codingSeq) - length($codingSeq2) );
      say "number of splice things: $numSpliceStuff";
      say "can be explained by splice things?: " . 
        (length($codingSeq) - length($codingSeq2) ) eq $numSpliceStuff ? "Yes" : "No";
    }

    push @errors, 'coding sequence calcualted by exclusion of annotated sites not equal to the one built from exon position intersection with coding sequence';
  }

  if ( length($codingSeq) % 3 ) {
    push @errors, 'coding sequence not divisible by 3';
  }

  if ( $codingSeq !~ m/$atgRe/ ) {
    push @errors, 'coding sequence doesn\'t start with ATG';
  }

  if ( $codingSeq !~ m/$stopCodonRe/ ) {
    push @errors, 'coding sequnce doesn\'t end with TAA, TAG, or TGA';
  }

  $self->_writeTranscriptErrors(\@errors);
  return \@errors;
}

__PACKAGE__->meta->make_immutable;

1;