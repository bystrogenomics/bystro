use 5.10.0;
use strict;
use warnings;

package Seq::Output::Fields;
use Mouse::Role 2;

has chromField => (is => 'ro', default => 'chrom', lazy => 1);
has posField => (is => 'ro', default => 'pos', lazy => 1);
has typeField => (is => 'ro', default => 'type', lazy => 1);
has discordantField => (is => 'ro', default => 'discordant', lazy => 1);
has altField => (is => 'ro', default => 'alt', lazy => 1);
has trTvField => (is => 'ro', default => 'trTv', lazy => 1);
has heterozygotesField => (is => 'ro', default => 'heterozygotes', lazy => 1);
has heterozygosityField => (is => 'ro', default => 'heterozygosity', lazy => 1);
has homozygotesField => (is => 'ro', default => 'homozygotes', lazy => 1);
has homozygosityField => (is => 'ro', default => 'homozygosity', lazy => 1);
has missingField => (is => 'ro', default => 'missingGenos', lazy => 1);
has missingnessField => (is => 'ro', default => 'missingness', lazy => 1);
has sampleMafField => (is => 'ro', default => 'sampleMaf', lazy => 1);
has idField => (is => 'ro', default => 'id', lazy => 1);
has acField => (is => 'ro', default => 'ac', lazy => 1);
has anField => (is => 'ro', default => 'an', lazy => 1);
has vcfPosField => (is => 'ro', default => 'vcfPos', lazy => 1);

no Mouse::Role;
1;