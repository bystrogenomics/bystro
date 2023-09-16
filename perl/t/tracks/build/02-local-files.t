use 5.10.0;
use strict;
use warnings;

use lib './lib';
use Seq::Tracks::Build::LocalFilesPaths;
use Path::Tiny;
use Test::More;

my $localPaths = Seq::Tracks::Build::LocalFilesPaths->new();


my $trackName = 'something';
my $filesDir = 'a_files_dir';

my $localFiles = [
  'something.chr*.txt',
  'something.shouldnt_match_glob.chr99.txt',
];

my @chrs = ('chr1', 'chr2', 'chr3', 'chr4', 'chr5');

my $path = path($filesDir)->child($trackName);
$path->mkpath();

my @actualPaths;
for my $chr (@chrs) {
  my $filePath = $path->child("something.$chr.txt")->absolute();

  $filePath->touch();

  push @actualPaths, $filePath->stringify;
}

my $nonGlobFile = $path->child("something.shouldnt_match_glob.chr99.txt")->absolute();
$nonGlobFile->touch();

push @actualPaths, $nonGlobFile;

my $computedPaths = $localPaths->makeAbsolutePaths($filesDir, $trackName, $localFiles);

for my $i ( 0 .. $#$computedPaths) {
  ok($computedPaths->[$i] eq $actualPaths[$i]);
}

$path->remove_tree();

done_testing();