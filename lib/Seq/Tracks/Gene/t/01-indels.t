use 5.10.0;
use strict;
use warnings;

use Data::Dump qw/ dump /;
use Path::Tiny;
use Test::More qw(no_plan);
use Test::Mouse::More;
use Test::Deep;
use Seq::Sites::Indels;
use Seq::KCManager;
use Test::MockObject;
use Test::MockObject::Extends;
use DDP;

validate_class 'Seq::Sites::Indels' => (
  attributes => [
    alleles => {
      is       => 'ro',
      isa      => 'Indels',
      coerce   => 1,
      traits   => ['Array'],
      handles  => { allAlleles => 'elements', },
      required => 1,
    },
  ],
  does      => ['Seq::Site::Gene::Definition'],
  methods   => [ 'findGeneData', '_annotateSugar' ],
  immutable => 1,
);

my $package = "Seq::Sites::Indels";

my $type  = 'Del';
my $frame = 'FrameShift';

# object creation
my $obj_frameshift = $package->new( { alleles => [-1] } );
ok( $obj_frameshift, 'object creation' );

my $dbm = Test::MockObject->new( sub { } );

$dbm->mock( 'db_bulk_get', \&build_coding_frameshift );

$obj_frameshift->findGeneData( 103620639, $dbm );

my @annotations     = $obj_frameshift->allAlleles;
my $annotation_type = $annotations[0]->annotation_type;
my $minor_allele    = $annotations[0]->minor_allele;
is( $minor_allele,    '-C' );
is( $annotation_type, "$type-$frame" . "[Coding]" );

$dbm->mock( 'db_bulk_get', \&build_startloss_frameshift );
$obj_frameshift->findGeneData( 103620639, $dbm );

@annotations     = $obj_frameshift->allAlleles;
$annotation_type = $annotations[0]->annotation_type;
$minor_allele    = $annotations[0]->minor_allele;
is( $minor_allele, '-G' );
is( $annotation_type, "$type-$frame" . "[StartLoss]", $annotation_type );

$dbm->mock( 'db_bulk_get', \&build_stoploss_frameshift );
$obj_frameshift->findGeneData( 103620639, $dbm );

@annotations     = $obj_frameshift->allAlleles;
$annotation_type = $annotations[0]->annotation_type;
$minor_allele    = $annotations[0]->minor_allele;
is( $minor_allele, '-G' );
is( $annotation_type, "$type-$frame" . "[StopLoss]", $annotation_type );

$dbm->mock( 'db_bulk_get', \&build_noncoding_frameshift );
$obj_frameshift->findGeneData( 103620639, $dbm );

@annotations     = $obj_frameshift->allAlleles;
$annotation_type = $annotations[0]->annotation_type;
$minor_allele    = $annotations[0]->minor_allele;
is( $minor_allele, '-G' );
is( $annotation_type, "$type" . "[3UTR]", $annotation_type );

###infrmae
my $obj_inframe = $package->new( { alleles => [-3] } );
ok( $obj_inframe, 'object creation' );

$frame = 'InFrame';
$dbm->mock( 'db_bulk_get', \&build_coding_inframe );

$obj_inframe->findGeneData( 103620639, $dbm );

@annotations     = $obj_inframe->allAlleles;
$annotation_type = $annotations[0]->annotation_type;
$minor_allele    = $annotations[0]->minor_allele;
is( $minor_allele, '-CGG' );
is( $annotation_type, "$type-$frame" . "[Coding|Coding|Coding]", $annotation_type );

$dbm->mock( 'db_bulk_get', \&build_startloss_inframe );
$obj_inframe->findGeneData( 103620639, $dbm );

@annotations     = $obj_inframe->allAlleles;
$annotation_type = $annotations[0]->annotation_type;
$minor_allele    = $annotations[0]->minor_allele;
is( $minor_allele, '-GTA' );
is( $annotation_type, "$type-$frame" . "[StartLoss|StartLoss|StartLoss]", $annotation_type );

$dbm->mock( 'db_bulk_get', \&build_stoploss_inframe );
$obj_inframe->findGeneData( 103620639, $dbm );

@annotations     = $obj_inframe->allAlleles;
$annotation_type = $annotations[0]->annotation_type;
$minor_allele    = $annotations[0]->minor_allele;
is( $minor_allele, '-GAT' );
is( $annotation_type, "$type-$frame" . "[StopLoss|StopLoss|StopLoss]", $annotation_type );

$dbm->mock( 'db_bulk_get', \&build_noncoding_inframe );
$obj_inframe->findGeneData( 103620639, $dbm );

@annotations     = $obj_inframe->allAlleles;
$annotation_type = $annotations[0]->annotation_type;
$minor_allele    = $annotations[0]->minor_allele;
is( $minor_allele, '-GAT' );
is( $annotation_type, "$type" . "[3UTR|3UTR|3UTR]", $annotation_type );

$dbm->mock( 'db_bulk_get', \&build_coding_noncoding_inframe );
$obj_inframe->findGeneData( 103620639, $dbm );

$annotation_type = $annotations[0]->annotation_type;
$minor_allele    = $annotations[0]->minor_allele;
is( $minor_allele, '-GAT' );
# is($annotation_type, "$type-$frame"."[3UTR|StopLoss]", $annotation_type);
is(
  $annotation_type,
  "$type-$frame" . "[3UTR|3UTR|StopLoss]",
  'Deletion ordered descending'
);

########################## insertions ##############################
$type  = 'Ins';
$frame = 'InFrame';
my $obj_ins_inframe = $package->new( { alleles => ["+AGT"] } );
ok( $obj_ins_inframe, 'insertion allele creation' );

$dbm->mock( 'db_bulk_get', \&build_coding_splice_insertion );

$obj_ins_inframe->findGeneData( 103620639, $dbm );
@annotations     = $obj_ins_inframe->allAlleles;
$annotation_type = $annotations[0]->annotation_type;
$minor_allele    = $annotations[0]->minor_allele;
is( $minor_allele, '+AGT' );
is( $annotation_type, "$type-$frame" . "[Coding|Splice Acceptor]",
  $annotation_type );

$dbm->mock( 'db_bulk_get', \&build_stop_utr_insertion );

$obj_ins_inframe->findGeneData( 103620639, $dbm );
@annotations     = $obj_ins_inframe->allAlleles;
$annotation_type = $annotations[0]->annotation_type;
$minor_allele    = $annotations[0]->minor_allele;
is( $minor_allele, '+AGT' );
is( $annotation_type, "$type-$frame" . "[StopLoss|3UTR]", $annotation_type );

$dbm->mock( 'db_bulk_get', \&build_noncoding_insertion );

$obj_ins_inframe->findGeneData( 103620639, $dbm );
@annotations     = $obj_ins_inframe->allAlleles;
$annotation_type = $annotations[0]->annotation_type;
$minor_allele    = $annotations[0]->minor_allele;
is( $minor_allele, '+AGT' );
is( $annotation_type, "$type" . "[3UTR|3UTR]", $annotation_type );

my $obj_ins_frameshift = $package->new( { alleles => ["+A"] } );
ok( $obj_ins_inframe, 'object creation' );

###############################################################################
# sub routines
###############################################################################

sub build_coding_frameshift {
  return (
    [
      {
        abs_pos        => 103620639,
        ref_base       => "C",
        ref_codon_seq  => "GGC",
        codon_number   => 278,
        codon_position => 2,
        ref_aa_residue => "G",
        site_type      => "Coding",
      }
    ]
  );
}

sub build_coding_inframe {
  my @arr = (
    [
      {
        abs_pos        => 103620639,
        ref_codon_seq  => "GGC",
        codon_number   => 278,
        codon_position => 2,
        ref_aa_residue => "G",
        ref_base       => "C",
        site_type      => "Coding",
      },
    ],
    [
      {
        abs_pos        => 103620638,
        ref_codon_seq  => "GGC",
        codon_number   => 278,
        codon_position => 1,
        ref_aa_residue => "G",
        ref_base       => "G",
        site_type      => "Coding",
      },
    ],
    [
      {
        abs_pos        => 103620637,
        ref_codon_seq  => "GGC",
        codon_number   => 278,
        codon_position => 0,
        ref_aa_residue => "G",
        ref_base       => "G",
        site_type      => "Coding",
      }
    ],
  );
  return @arr;
}

sub build_startloss_frameshift {
  return (
    [
      {
        abs_pos        => 103620639,
        codon_number   => 1,
        codon_position => 2,
        ref_aa_residue => "M",
        ref_base       => "G",
        ref_codon_seq  => "ATG",
        site_type      => "Coding",
      }
    ]
  );
}

sub build_startloss_inframe {
  my @arr = (
    [
      {
        abs_pos        => 103620639,
        codon_number   => 1,
        codon_position => 2,
        ref_aa_residue => "M",
        ref_base       => "G",
        ref_codon_seq  => "ATG",
        site_type      => "Coding",
      },
    ],
    [
      {
        abs_pos        => 103620638,
        codon_number   => 1,
        codon_position => 1,
        ref_aa_residue => "M",
        ref_base       => "T",
        ref_codon_seq  => "ATG",
        site_type      => "Coding",
      },
    ],
    [
      {
        abs_pos        => 103620637,
        codon_number   => 1,
        codon_position => 0,
        ref_aa_residue => "M",
        ref_base       => "A",
        ref_codon_seq  => "ATG",
        site_type      => "Coding",
      },
    ],
  );
  return @arr;
}

sub build_stoploss_frameshift {
  return (
    [
      {
        abs_pos        => 103620639,
        codon_number   => 300,      #fake, just not 1
        codon_position => 2,
        ref_aa_residue => "*",
        ref_base       => "G",
        ref_codon_seq  => "TAG",
        site_type      => "Coding",
      }
    ]
  );
}

sub build_stoploss_inframe {
  my @arr = (
    [
      {
        abs_pos        => 103620639,
        codon_number   => 300,      #fake, just not 1
        codon_position => 2,
        ref_aa_residue => "*",
        ref_base       => "G",
        ref_codon_seq  => "TAG",
        site_type      => "Coding",
      },
    ],
    [
      {
        abs_pos        => 103620638,
        codon_number   => 300,
        codon_position => 1,
        ref_aa_residue => "*",
        ref_base       => "A",
        ref_codon_seq  => "TAG",
        site_type      => "Coding",
      },
    ],
    [
      {
        abs_pos        => 103620637,
        codon_number   => 300,
        codon_position => 0,
        ref_aa_residue => "*",
        ref_base       => "T",
        ref_codon_seq  => "TAG",
        site_type      => "Coding",
      }
    ],
  );
  return @arr;
}

sub build_noncoding_frameshift {
  return (
    [
      {
        abs_pos   => 103620639,
        ref_base  => "G",
        site_type => "3UTR",
      }
    ],
  );
}

sub build_noncoding_inframe {
  my @arr = (
    [
      {
        abs_pos   => 103620639,
        ref_base  => "G",
        site_type => "3UTR",
      },
    ],
    [
      {
        abs_pos   => 103620638,
        ref_base  => "A",
        site_type => "3UTR",
      },
    ],
    [
      {
        abs_pos   => 103620637,
        ref_base  => "T",
        site_type => "3UTR",
      }
    ],
  );
  return @arr;
}

sub build_coding_noncoding_inframe {
  my @arr = (
    [
      {
        abs_pos   => 103620639,
        ref_base  => "G",
        site_type => "3UTR",
      },
    ],
    [
      {
        abs_pos   => 103620638,
        ref_base  => "A",
        site_type => "3UTR",
      },
    ],
    [
      {
        abs_pos        => 103620637,
        codon_number   => 300,
        codon_position => 0,
        ref_aa_residue => "*",
        ref_base       => "T",
        ref_codon_seq  => "TAG",
        site_type      => "Coding",
      }
    ],
  );
  return @arr;
}

sub build_noncoding_insertion {
  my @arr = (
    [
      {
        abs_pos   => 103620638,
        ref_base  => "A",
        site_type => "3UTR",
      },
    ],
    [
      {
        abs_pos   => 103620639,
        ref_base  => "G",
        site_type => "3UTR",
      },
    ],
  );
  return @arr;
}

sub build_stop_utr_insertion {
  my @arr = (
    [
      {
        abs_pos        => 103620638,
        codon_number   => 300,
        codon_position => 1,
        ref_aa_residue => "*",
        ref_base       => "A",
        ref_codon_seq  => "TAG",
        site_type      => "Coding",
      },
    ],
    [
      {
        abs_pos   => 103620639,
        ref_base  => "G",
        site_type => "3UTR",
      },
    ],
  );
  return @arr;
}

sub build_coding_splice_insertion {
  my @arr = (
    [
      {
        abs_pos        => 103620638,
        codon_number   => 300,
        codon_position => 2,
        ref_aa_residue => "L",
        ref_base       => "G",
        ref_codon_seq  => "TAG",
        site_type      => "Coding",
      },
    ],
    [
      {
        abs_pos   => 103620639,
        ref_base  => "G",
        site_type => "Splice Acceptor",
      },
    ],
  );
  return @arr;
}

sub check_isa {
  my $class   = shift;
  my $parents = shift;

  local $Test::Builder::Level = $Test::Builder::Level + 1;

  my @isa = $class->meta->linearized_isa;
  shift @isa; # returns $class as the first entry

  my $count = scalar @{$parents};
  my $noun = PL_N( 'parent', $count );

  is( scalar @isa, $count, "$class has $count $noun" );

  for ( my $i = 0; $i < @{$parents}; $i++ ) {
    is( $isa[$i], $parents->[$i], "parent[$i] is $parents->[$i]" );
  }
}
