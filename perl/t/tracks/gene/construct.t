# Test to see if we can construct a gene object
use Test::More;

use Seq::Tracks::Gene::Build;
use Seq::DBManager

  Seq::DBManager::initialize( { databaseDir => 'bar' } );

my $gene = Seq::Tracks::Gene::Build->new(
  {
    files_dir   => 'foo',
    name        => 'refSeqTrack',
    type        => 'gene',
    assembly    => 'hg19',
    chromosomes => [ 'chr1', 'chr2' ],
  }
);

ok( $gene->isa('Seq::Tracks::Gene::Build'), 'Gene object created' );

done_testing();
