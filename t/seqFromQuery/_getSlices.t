use 5.10.0;
use strict;
use warnings;

use lib './lib';
use SeqFromQuery;
use Path::Tiny;
use Test::More;
use DDP;

my $config = {
    inputQueryBody => {prop => 'stuff'},
    indexName => 'test',
    indexType => 'test',
    assembly => 'hg38',
    configPath => './',
    fieldNames => ['1','2','3'],
    indexConfig => {some => 1, one => 2},
    connection => {someProp => 1},
    database_dir => 'some_fake',
    output_file_base => './',
    tracks => {
        tracks =>
            [{
                name => 'someTrack',
                type => 'gene'
            }]
    
    }
};

$config->{maxThreads} = 8;
$config->{indexConfig}{index_settings}->{index}->{number_of_shards} = 6;

my $seq = SeqFromQuery->new($config);

my $slices = $seq->_getSlices();

ok($slices == 12);

$config->{maxThreads} = 16;
$config->{indexConfig}{index_settings}->{index}->{number_of_shards} = 5;

$seq = SeqFromQuery->new($config);

$slices = $seq->_getSlices();

ok ($slices == 5 * 4);

$config->{maxThreads} = 16;
$config->{indexConfig}{index_settings}->{index}->{number_of_shards} = 6;

$seq = SeqFromQuery->new($config);

$slices = $seq->_getSlices();

ok ($slices == 6 * 2);

$config->{maxThreads} = 13;
$config->{indexConfig}{index_settings}->{index}->{number_of_shards} = 3;

$seq = SeqFromQuery->new($config);

$slices = $seq->_getSlices();

ok ($slices == 3 * 4);

$config->{maxThreads} = 12;
$config->{indexConfig}{index_settings}->{index}->{number_of_shards} = 6;

$seq = SeqFromQuery->new($config);

$slices = $seq->_getSlices();

ok ($slices == 6 * 2);

$config->{maxThreads} = 12;
$config->{indexConfig}{index_settings}->{index}->{number_of_shards} = 16;

my $seq2 = SeqFromQuery->new($config);

$slices = $seq2->_getSlices();

ok ($slices == 16 * 2);


done_testing();