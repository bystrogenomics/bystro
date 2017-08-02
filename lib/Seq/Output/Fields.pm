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
has homozygotesField => (is => 'ro', default => 'homozygotes', lazy => 1);
has missingField => (is => 'ro', default => 'missingGenos', lazy => 1);

no Mouse::Role;
1;