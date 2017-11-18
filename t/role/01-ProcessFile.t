use strict;
use warnings;
use 5.10.0;

use Path::Tiny;
use Test::More tests => 12;

sub _build_headers {
  return {
    snp_1 => [qw/ Fragment Position Reference Minor_allele Type /],
    snp_2 => [qw/ Fragment Position Reference Alleles Allele_Counts Type /],
  };
}

sub _build_headers_qw {
  return {
    snp_1 => qw/ Fragment Position Reference Minor_allele Type /,
    snp_2 => qw/ Fragment Position Reference Alleles Allele_Counts Type /,
  };
}

my $headersHref = _build_headers();
my $qwHeadersHref = _build_headers_qw();

ok(ref $headersHref->{snp_1} eq 'ARRAY', 'header ref is ARRAY');
ok(ref $qwHeadersHref->{snp_1} eq '', 'qw returns scalar' );
ok($headersHref->{snp_1}[0] eq 'Fragment', 
  'header array ref first element is Fragment'); 
ok($qwHeadersHref->{snp_1} eq 'Fragment', 
  'qw evaluated to first element in scalar context, which is Fragment');

my $err;

ok(int(!$err) == 1, 'Undefined boolean compliment is truthy');

$err = 'Some error';

ok(int(!$err) == 0, 'String boolean compliment is falsey');

ok(@{$headersHref->{snp_2} } == 6, 'Can calculate array ref length');

ok($#{$headersHref->{snp_2} } == 5, 'Can dereference array ref and get last index');

ok($#{$headersHref->{snp_1} } == 4, 'Can dereference array ref and get last index 2');


ok(@{$headersHref->{snp_2} }[1] eq 'Position', 'Can deference array ref');

my $name = path("foo.txt")->basename(qr/\.\w*/); 
my $name2 = path("foo.bar.baz.txt")->basename(qr/\.\w*/); 

is($name, 'foo');
is($name2,'foo.bar.baz');


