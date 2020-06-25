use 5.10.0;
use strict;
use warnings;
# use blib;
use Benchmark qw(cmpthese :hireswallclock);
use Sereal::Decoder qw(decode_sereal sereal_decode_with_object);
use Sereal::Encoder qw(encode_sereal sereal_encode_with_object);
use Storable qw(nfreeze thaw);
use Data::Dumper qw(Dumper);
use Math::SigFigs qw(:all);
use DDP;
use Data::MessagePack;
use CBOR::XS qw(encode_cbor decode_cbor);
use Getopt::Long qw(GetOptions);
use Compress::Snappy qw/compress decompress/;
require bytes;

use lib './lib';
use Seq::DBManager;
use Scalar::Util qw/looks_like_number/;
use Data::Float;
# my $seq = MockBuilder->new_with_config({config => './config/hg19.yml', debug => 1});
Seq::DBManager::initialize(
    databaseDir => '/mnt/annotator/bystro-dev/hg19/index',
    readOnly => 1,
);
my $db = Seq::DBManager->new();
# my $stuff = $db->dbReadOne('chr21', 49e6, 0 , 1);
# p $stuff;
my $nobless;

my $mpt = Data::MessagePack->new()->prefer_float32();

my $test = encode_cbor(-24);
# say "length of 64 is  " . (length($test));
# exit;
#Credit to https://github.com/Sereal/Sereal/wiki/Sereal-Comparison-Graphs

GetOptions(
    'secs|duration=f'                    => \( my $duration           = -5 ),
    'encoder'                            => \( my $encoder            = 0 ),
    'decoder'                            => \( my $decoder            = 0 ),
    'dump|d'                             => \( my $dump               = 0 ),
    'only=s@'                            => \( my $only               = undef ),
    'exclude=s@'                         => \( my $exclude            = undef ),
    'tiny'                               => \( my $tiny_data          = 0 ),
    'small'                              => \( my $small_data         = 0 ),
    'medium'                             => \( my $medium_data        = 0 ),
    'large'                              => \( my $large_data         = 0 ),
    'very_large|very-large|verylarge'    => \( my $very_large_data    = 0 ),
    'array'    => \( my $array_data    = 0 ),
    'small_array'    => \( my $small_array    = 0 ),
    'large_array' => \(my $large_array = 0),
    'large_array_sigfig' => \(my $large_array_sigfig = 0),
    'real_array' => \(my $real_array = 0),
    'real_array_sigfig' => \(my $real_array_sigfig = 0),
    'small_hash' => \(my $small_hash = 0),
    # 'no_bless|no-bless|nobless'          => \( my $nobless            = 0 ),
    'sereal_only|sereal-only|serealonly' => \( my $sereal_only        = 0 ),
    'diagrams'                           => \( my $diagrams           = 0 ),
    'diagram_output=s'                   => \( my $diagram_output_dir = "" ),
) or die "Bad option";

my $fail =
  $tiny_data + $small_data + $medium_data + $very_large_data + $large_data - 1;
if ( $fail and $fail > 0 ) {
    die "Only one of --tiny, --small, --medium, --large, --very-large allowed!";
}
$encoder = 1 if not $encoder and not $decoder;

#our %opt = @ARGV;
our %opt;

my $data_set_name;
srand(0);
my $chars = join( "", "a" .. "z", "A" .. "Z" ) x 2;
my @str;
push @str, substr( $chars, int( rand( int( length($chars) / 2 + 1 ) ) ), 10 )
  for 1 .. 1000;
my @rand = map rand, 1 .. 1000;

our (
    $enc, $dec,
    $enc_snappy,        $dec_snappy,
    $enc_zlib_fast,     $dec_zlib_fast,
    $enc_zlib_small,    $dec_zlib_small,
    $jsonxs, $msgpack, $dd_noindent, $dd_indent, $cbor
);
my $storable_tag= "strbl";
my $sereal_tag= "srl";
my %meta = (
    # jxs => {
    #     enc  => '$::jsonxs->encode($data);',
    #     dec  => '$::jsonxs->decode($encoded);',
    #     name => 'JSON::XS OO',
    #     init => sub {
    #         $jsonxs = JSON::XS->new()->allow_nonref();
    #     },
    #     use => 'use JSON::XS qw(decode_json encode_json);',
    # },
    # ddl => {
    #     enc  => 'DumpLimited($data);',
    #     dec  => 'Data::Undump::undump($encoded);',
    #     name => 'Data::Dump::Limited',
    #     use  => [
    #                 'use Data::Undump qw(undump);',
    #                 'use Data::Dumper::Limited qw(DumpLimited);',
    #             ],
    # },
    mp_float32 => {
        enc  => '$::msgpack->pack($data);',
        dec  => '$::msgpack->unpack($encoded);',
        name => 'Data::MsgPack',
        use  => 'use Data::MessagePack;',
        init => sub {
            $msgpack = Data::MessagePack->new()->prefer_integer()->prefer_float32();
        },
    },
    mp_float32_snappy => {
        enc  => 'compress($::msgpack->pack($data));',
        dec  => '$::msgpack->unpack(decompress($encoded));',
        name => 'Data::MsgPack',
        use  => 'use Data::MessagePack;',
        init => sub {
            $msgpack = Data::MessagePack->new()->prefer_integer()->prefer_float32();
        },
    },
    # mp => {
    #     enc  => '$::msgpack->pack($data);',
    #     dec  => '$::msgpack->unpack($encoded);',
    #     name => 'Data::MsgPack',
    #     use  => 'use Data::MessagePack;',
    #     init => sub {
    #         $msgpack = Data::MessagePack->new();
    #     },
    # },
    cbor => {
        enc  => 'encode_cbor($data);',
        dec  => 'decode_cbor($encoded);',
        name => 'CBOR::XS',
        use => 'use CBOR::XS qw(encode_cbor decode_cbor);',
    },
    cbor_snappy => {
        enc  => 'compress(encode_cbor($data));',
        dec  => 'decode_cbor(decompress($encoded));',
        name => 'CBOR::XS',
        use => 'use CBOR::XS qw(encode_cbor decode_cbor);',
        init => sub {
            $cbor= CBOR::XS->new();
        },
    },
    # dd_noind => {
    #     enc  => 'Data::Dumper->new([$data])->Indent(0)->Dump();',
    #     dec  => 'eval $encoded;',
    #     name => 'Data::Dumper no-indent',
    # },
    # dd => {
    #     enc  => 'Dumper($data);',
    #     dec  => 'eval $encoded;',
    #     name => 'Data::Dumper indented',
    # },
    # $storable_tag => {
    #     enc  => 'nfreeze($data);',
    #     dec  => 'thaw($encoded);',
    #     name => 'Storable',
    # },
    # srl_func => {
    #     enc  => 'encode_sereal($data, $opt);',
    #     dec  => 'decode_sereal($encoded, $opt);',
    #     name => 'Sereal functional',
    # },
    # srl_fwo => {
    #     enc  => 'sereal_encode_with_object($::enc,$data);',
    #     dec  => 'sereal_decode_with_object($::dec,$encoded);',
    #     name => 'Sereal functional with object',
    # },
    # $sereal_tag => {
    #     enc  => '$::enc->encode($data);',
    #     dec  => '$::dec->decode($encoded);',
    #     name => 'Sereal OO',
    #     init => sub {
    #         $enc = Sereal::Encoder->new( %opt ? \%opt : () );
    #         $dec = Sereal::Decoder->new( \%opt ? \%opt : () );
    #     },
    # },
    # srl_snpy => {
    #     enc  => '$::enc_snappy->encode($data);',
    #     dec  => '$::dec_snappy->decode($encoded);',
    #     name => 'Sereal OO snappy',
    #     init => sub {
    #         $enc_snappy = Sereal::Encoder->new(
    #             {
    #                 %opt,
    #                 compress => Sereal::Encoder::SRL_SNAPPY
    #             }
    #         );
    #         $dec_snappy = Sereal::Decoder->new( %opt ? \%opt : () );
    #     },
    # },
    # srl_zfast => {
    #     enc  => '$::enc_zlib_fast->encode($data);',
    #     dec  => '$::dec_zlib_fast->decode($encoded);',
    #     name => 'Sereal OO zlib fast',
    #     init => sub {
    #         $enc_zlib_fast = Sereal::Encoder->new(
    #             {
    #                 %opt,
    #                 compress           => Sereal::Encoder::SRL_ZLIB,
    #                 compress_level     => 6,
    #                 compress_threshold => 0,
    #             }
    #         );
    #         $dec_zlib_fast = Sereal::Decoder->new( %opt ? \%opt : () );
    #     },
    # },
    # srl_zbest => {
    #     enc  => '$::enc_zlib_small->encode($data);',
    #     dec  => '$::dec_zlib_small->decode($encoded);',
    #     name => 'Sereal OO zib best',
    #     init => sub {
    #         $enc_zlib_small = Sereal::Encoder->new(
    #             {
    #                 %opt,
    #                 compress           => Sereal::Encoder::SRL_ZLIB,
    #                 compress_level     => 10,
    #                 compress_threshold => 0,
    #             }
    #         );
    #         $dec_zlib_small = Sereal::Decoder->new( %opt ? \%opt : () );
    #     },
    # },
);
if ($only) {
    my @pat= map { split /\s*,\s*/, $_ } @$only;
    $only = {};
    foreach my $key (keys %meta) {
        $key=~/$_/ and $only->{$key}= 1
            for @pat;
    }
    die "Only [@pat] produced no matches!" unless keys %$only;
}
if ($exclude) {
    my @pat= map { split /\s*,\s*/, $_ } @$exclude;
    $exclude = {};
    foreach my $key (keys %meta) {
        $key=~/$_/ and $exclude->{$key}= 1
            for @pat;
    }
    die "Exclude [@pat] produced no matches!" unless keys %$exclude;
}

our %data;
our %encoded;
our %decoded;
our %enc_bench;
our %dec_bench;
our %sizes;
my $dbLength = $db->dbGetNumberOfEntries('chr21');
p $dbLength;


for my $i (16.2e6 .. 17e6) {
    my $dbData = $db->dbReadOne('chr21', $i);

    if(!$dbData) {
        next;
    }

    foreach my $key ( sort keys %meta ) {
        my $info = $meta{$key};
        $info->{tag}= $key;
        next if $only    and not $only->{$key}    and $key ne $storable_tag;
        next if $exclude and     $exclude->{$key} and $key ne $storable_tag;
        if (my $use= $info->{use}) {
            $use= [$use] unless ref $use;
            $use= join ";\n", @$use, 1;
            unless (eval $use) {
                warn "Can't load dependencies for $info->{name}, skipping\n";
                next;
            }
        }
        $info->{enc}=~s/\$data/\$::data{$key}/g;
        $info->{dec}=~s/\$encoded/\$::encoded{$key}/g;
        $info->{enc}=~s/\$opt/%opt ? "\\%::opt" : ""/ge;
        $info->{dec}=~s/\$opt/%opt ? "\\%::opt" : ""/ge;

        $data{$key}    = $dbData;
        $info->{init}->() if $info->{init};
        $encoded{$key} = eval $info->{enc}
          or die "Failed to eval $info->{enc}: $@";
        $decoded{$key} = eval '$::x = ' . $info->{dec} . '; 1'
          or die "Failed to eval $info->{dec}: $@\n$encoded{$key}\n";
        $info->{size}    = bytes::length( $encoded{$key} );
        next if $only    and not $only->{$key};
        next if $exclude and     $exclude->{$key};
        $enc_bench{$key} = '$::x_' . $key . ' = ' . $info->{enc};
        $dec_bench{$key} = '$::x_' . $key . ' = ' . $info->{dec};
        $sizes{$info->{tag}} += $info->{size};
    }

    # my $sereal = $encoded{$sereal_tag};
    # print($sereal), exit if $dump;

    # my $storable_len = bytes::length($encoded{$storable_tag});
    # foreach my $info (
    #     sort { $a->{size} <=> $b->{size} || $a->{name} cmp $b->{name} }
    #     grep { defined $_->{size} }
    #     values %meta
    # ) {
    #     next unless $info->{size};
    #     if ($info->{tag} eq $storable_tag) {
    #         printf "%-40s %12d bytes\n",
    #             $info->{name} . " ($info->{tag})", $info->{size};
    #     } else {
    #         printf "%-40s %12d bytes %6.2f%% of $storable_tag\n",
    #             $info->{name} . " ($info->{tag})", $info->{size},
    #             $info->{size} / $storable_len * 100;
    #     }
    # }
}

my @keys = sort {$sizes{$a} <=> $sizes{$b}} keys %sizes;

my @results;
for my $key (@keys) {
    push @results, [$key, $sizes{$key}];
}

p @results;
p %sizes;

our $x;
my ( $encoder_result, $decoder_result );
if ($encoder) {
    print "\n* Timing encoders\n";
    $encoder_result = cmpthese( $duration, \%enc_bench );
}

if ($decoder) {
    print "\n* Timing decoders\n";
    $decoder_result = cmpthese( $duration, \%dec_bench );
}


sub make_data {
    if ($tiny_data) {
        $data_set_name = "empty hash";
        return {};
    }
    elsif ($small_data) {
        $data_set_name = "small hash";
        return {
            foo => 1,
            bar => [ 100, 101, 102 ],
            float => 1.32023e-6,
            str => "this is a \x{df} string which has to be serialized"
        };
    }
    elsif ($medium_data) {
        my @obj = (
            {
                foo => 1,
                bar => [ 100, 101, 102 ],
                float => .000152,
                str => "this is a \x{df} string which has to be serialized"
            },
            {
                foo => 2,
                bar => [ 103, 103, 106, 999 ],
                float => 23.234,
                str2 =>
                  "this is a \x{df} aaaaaastring which has to be serialized"
            },
            {
                foozle => 3,
                bar    => [100],
                float => .02,
                str3 =>
                  "this is a \x{df} string which haaaaadsadas to be serialized"
            },
            {
                foozle => 3,
                bar    => [],
                float => .00001,
                st4r =>
                  "this is a \x{df} string which has to be sdassdaerialized"
            },
            {
                foo  => 1,
                bar  => [ 100, 101, 102 ],
                float => -1.2,
                s5tr => "this is a \x{df} string which has to be serialized"
            },
            {
                foo => 2,
                bar => [ 103, 103, 106, 999 ],
                float => 45.23,
                str =>
                  "this is a \x{df} aaaaaastring which has to be serialized"
            },
            {
                foozle => 3,
                bar    => [100],
                float => .00000012,
                str =>
                  "this is a \x{df} string which haaaaadsadas to be serialized"
            },
            {
                foozle => 3,
                bar    => [],
                float => .00000052,
                str2 =>
                  "this is a \x{df} string which has to be sdassdaerialized"
            },
            {
                foo2 => -99999,
                bar  => [ 100, 101, 102 ],
                float => 15.32,
                str2 => "this is a \x{df} string which has to be serialized"
            },
            {
                foo2 => 213,
                bar  => [ 103, 103, 106, 999 ],
                float => 20.02,
                str =>
                  "this is a \x{df} aaaaaastring which has to be serialized"
            },
            {
                foozle2 => undef,
                bar     => [100],
                float => .00001,
                str =>
                  "this is a \x{df} string which haaaaadsadas to be serialized"
            },
            {
                foozle2 => undef,
                bar     => [ 1 .. 20 ],
                float => 20.21,
                str =>
                  "this is a \x{df} string which has to be sdassdaerialized"
            },
        );
        my @classes = qw(Baz Baz Baz3 Baz2 Baz Baz Baz3 Baz2 Baz Baz Baz3 Baz2);
        if ( $nobless ) {
            $data_set_name = "array of small hashes with relations";
        }
        else {
            bless( $obj[$_], $classes[$_] ) for 0 .. $#obj;
            $data_set_name = "array of small objects with relations";
        }
        foreach my $i ( 1 .. $#obj ) {
            $obj[$i]->{parent} = $obj[ $i - 1 ];
        }
        return \@obj;
    }
    elsif ($very_large_data) {    # "large data"
        $data_set_name = "really rather large data structure";
        my @refs = (
            [ 1 .. 10000 ],
            {@str}, {@str}, [ 1 .. 10000 ],
            {@str}, [@rand], {@str}, {@str},
        );
        return [
            \@refs, \@refs,
            [ map { [ reverse 1 .. 100 ] } ( 0 .. 1000 ) ],
            [ map { +{ foo => "bar", baz => "buz" } } 1 .. 2000 ]
        ];
    } elsif ($small_array) {
      return [1, 2, 3, 'string', 5.018];
    } elsif ($array_data) {
        $data_set_name = "Bystro-like data composed of multiple values in an array";
        # my $data = [
        #     0, 23.2, 0.01, -1.0, [0, 1], ["Segawa syndrome, autosomal recessive","germline","Uncertain significance", 1, "G", "A", "criteria provided, single submitter"], [3.23625e-05, 5.86098e-05, 0, 0, 0, 0, 0, 30900, 14968, 302, 978, 3492, 17062, 13838], ["rs139474171", ["A", "G"], ["near-gene-5", "missense"], [0.000398, 0.999602], [49, 123117], "single", "+"]
        # ];
        my $data = [
     1,
     undef,
     [
         1710,
         13
    ],
     0,
     FormatSigFigs(0.14, 2) + 0,
     [
         FormatSigFigs(0.65, 2) + 0,
         FormatSigFigs(2.09, 2) + 0,
         FormatSigFigs(0.75, 2) + 0,
    ],
     [
         "rs77179864",
         "+",
         [
             "A",
             "G"
        ],
         "single",
         "intron",
         [
             "A",
             "G"
        ],
         [
             125137,
             361
        ],
         [
             FormatSigFigs(0.997123, 2) + 0,
             FormatSigFigs(0.002877, 2) + 0
        ]
    ],
     [
         132478,
         "Pheochromocytoma",
         "Conflicting interpretations of pathogenicity",
         "single nucleotide variant",
         "germline",
         2,
         "no assertion criteria provided",
         "C",
         "G"
    ],
     [
          "G",
          "rs77179864",
          FormatSigFigs(0.00264721, 2) + 0,
          30976,
          8728,
          838,
          302,
          1622,
          3492,
          15012,
         982,
         17122,
         13854,
         FormatSigFigs(0.00114574, 2) + 0,
         FormatSigFigs(0.00119332, 2) + 0,
         0,
         0,
         FormatSigFigs(0.000286369, 2) + 0,
         FormatSigFigs(0.00452971, 2) + 0,
         FormatSigFigs(0.00203666, 2) + 0,
         FormatSigFigs(0.00221937, 2) + 0,
         FormatSigFigs(0.00317598, 2) + 0
    ],
     [
          "G",
          "rs77179864",
          FormatSigFigs(0.00306743, 2) + 0,
          245156,
          15250,
          33010,
          9846,
          17200,
          22256,
          111556,
         5414,
         134338,
         110818,
         FormatSigFigs(0.00163934, 2) + 0,
         FormatSigFigs(0.00312027, 2) + 0,
         FormatSigFigs(0.000812513, 2) + 0,
         0,
         FormatSigFigs(0.000179727, 2) + 0,
         FormatSigFigs(0.00537846, 2) + 0,
        FormatSigFigs(0.00221648, 2) + 0,
         FormatSigFigs(0.00302223, 2) + 0,
         FormatSigFigs(0.00312224, 2) + 0
    ]
];

        return $data;
    } elsif($large_array) {
        $data_set_name = "Bystro-like large data composed of multiple values in an array, a few overlapping annotations per annotation type";
        # my $data = [
        #     0, 23.2, 0.01, -1.0, [0, 1], ["Segawa syndrome, autosomal recessive","germline","Uncertain significance", 1, "G", "A", "criteria provided, single submitter"], [3.23625e-05, 5.86098e-05, 0, 0, 0, 0, 0, 30900, 14968, 302, 978, 3492, 17062, 13838], ["rs139474171", ["A", "G"], ["near-gene-5", "missense"], [0.000398, 0.999602], [49, 123117], "single", "+"]
        # ];
        my $data = [
           2,
           undef,
           [
               1723,
               13
          ],
           0,
           0.01,
           [
               7.39,
               5.47,
               6.72
          ],
           [
               "rs2293083",
               "-",
               [
                   "A",
                   "C",
                   "G"
              ],
               "single",
               [
                   "intron",
                   "near-gene-5"
              ],
               [
                   "A",
                   "C",
                   "G"
              ],
               [
                   22,
                   14130,
                   5864
              ],
               [
                   0.001099,
                   0.705935,
                   0.292966
              ]
          ],
           undef,
           [
                [
                   "G",
                   "T"
              ],
                [
                   "rs2293083",
                   "rs2293083"
              ],
                [
                   0.696557,
                   0.000876908
              ],
                [
                   30790,
                   30790
              ],
                [
                   8672,
                   8672
              ],
                [
                   836,
                   836
              ],
                [
                   302,
                   302
              ],
                [
                   1622,
                   1622
              ],
                [
                   3492,
                   3492
              ],
                [
                   14892,
                   14892
              ],
               [
                   974,
                   974
              ],
               [
                   17012,
                   17012
              ],
               [
                   13778,
                   13778
              ],
               [
                   0.567804,
                   0.000461255
              ],
               [
                   0.726077,
                   0.00119617
              ],
               [
                   0.758278,
                   0
              ],
               [
                   0.82984,
                   0
              ],
               [
                   0.742554,
                   0.000286369
              ],
               [
                   0.740062,
                   0.001343
              ],
               [
                   0.746407,
                   0.00102669
              ],
               [
                   0.698389,
                   0.000940513
              ],
               [
                   0.694295,
                   0.000798374
              ]
          ],
           [
                [
                   "G",
                   "T",
                   "A"
              ],
                [
                   "rs2293083",
                   "rs2293083",
                   "rs2293083"
              ],
                [
                   0.732294,
                   0.0015717,
                   7.07975e-06
              ],
                [
                   141248,
                   141248,
                   141248
              ],
                [
                   6524,
                   6524,
                   6524
              ],
                [
                   23846,
                   23846,
                   23846
              ],
                [
                   8104,
                   8104,
                   8104
              ],
                [
                   10146,
                   10146,
                   10146
              ],
                [
                   14990,
                   14990,
                   14990
              ],
                [
                   51312,
                   51312,
                   51312
              ],
               [
                   3736,
                   3736,
                   3736
              ],
               [
                   76788,
                   76788,
                   76788
              ],
               [
                   64460,
                   64460,
                   64460
              ],
               [
                   0.559626,
                   0.000766401,
                   0
              ],
               [
                   0.734631,
                   0.00150969,
                   0
              ],
               [
                   0.752221,
                   0,
                   0
              ],
               [
                   0.832348,
                   0,
                   0
              ],
               [
                   0.732889,
                   0.00153436,
                   0
              ],
               [
                   0.740529,
                   0.0023971,
                   1.94886e-05
              ],
               [
                   0.755621,
                   0.00267666,
                   0
              ],
               [
                   0.731781,
                   0.00171902,
                   0
              ],
               [
                   0.732904,
                   0.00139621,
                   1.55135e-05
              ]
          ]
      ];

      return $data;
    } elsif($large_array_sigfig) {
        $data_set_name = "Bystro-like large data composed of multiple values in an array, a few overlapping annotations per annotation type";
        # my $data = [
        #     0, 23.2, 0.01, -1.0, [0, 1], ["Segawa syndrome, autosomal recessive","germline","Uncertain significance", 1, "G", "A", "criteria provided, single submitter"], [3.23625e-05, 5.86098e-05, 0, 0, 0, 0, 0, 30900, 14968, 302, 978, 3492, 17062, 13838], ["rs139474171", ["A", "G"], ["near-gene-5", "missense"], [0.000398, 0.999602], [49, 123117], "single", "+"]
        # ];
        my $data = [
           2,
           undef,
           [
               1723,
               13
          ],
           0,
           FormatSigFigs(0.01, 2) + 0,
           [
               FormatSigFigs(7.39, 2) + 0,
               FormatSigFigs(5.47, 2) + 0,
               FormatSigFigs(6.72, 2) + 0
          ],
           [
               "rs2293083",
               "-",
               [
                   "A",
                   "C",
                   "G"
              ],
               "single",
               [
                   "intron",
                   "near-gene-5"
              ],
               [
                   "A",
                   "C",
                   "G"
              ],
               [
                   22,
                   14130,
                   5864
              ],
               [
                   FormatSigFigs(0.001099, 2) + 0,
                   FormatSigFigs(0.705935, 2) + 0,
                   FormatSigFigs(0.292966, 2) + 0
              ]
          ],
           undef,
           [
                [
                   "G",
                   "T"
              ],
                [
                   "rs2293083",
                   "rs2293083"
              ],
                [
                   FormatSigFigs(0.696557, 2) + 0,
                   FormatSigFigs(0.000876908, 2) + 0
              ],
                [
                   30790,
                   30790
              ],
                [
                   8672,
                   8672
              ],
                [
                   836,
                   836
              ],
                [
                   302,
                   302
              ],
                [
                   1622,
                   1622
              ],
                [
                   3492,
                   3492
              ],
                [
                   14892,
                   14892
              ],
               [
                   974,
                   974
              ],
               [
                   17012,
                   17012
              ],
               [
                   13778,
                   13778
              ],
               [
                   FormatSigFigs(0.567804, 2) + 0,
                   FormatSigFigs(0.000461255, 2) + 0
              ],
               [
                   FormatSigFigs(0.726077,2) + 0,
                   FormatSigFigs(0.00119617,2) + 0
              ],
               [
                   FormatSigFigs(0.758278,2) + 0,
                   0
              ],
               [
                   FormatSigFigs(0.82984,2) + 0,
                   0
              ],
               [
                   FormatSigFigs(0.742554,2) + 0,
                   FormatSigFigs(0.000286369,2) + 0
              ],
               [
                   FormatSigFigs(0.740062,2) + 0,
                   FormatSigFigs(0.001343,2) + 0
              ],
               [
                   FormatSigFigs(0.746407,2) + 0,
                   FormatSigFigs(0.00102669,2) + 0
              ],
               [
                   FormatSigFigs(0.698389,2) + 0,
                   FormatSigFigs(0.000940513,2) + 0
              ],
               [
                   FormatSigFigs(0.694295,2) + 0,
                   FormatSigFigs(0.000798374,2) + 0
              ]
          ],
           [
                [
                   "G",
                   "T",
                   "A"
              ],
                [
                   "rs2293083",
                   "rs2293083",
                   "rs2293083"
              ],
                [
                   FormatSigFigs(0.732294,2) + 0,
                   FormatSigFigs(0.0015717,2) + 0,
                   FormatSigFigs(7.07975e-06,2) + 0
              ],
                [
                   141248,
                   141248,
                   141248
              ],
                [
                   6524,
                   6524,
                   6524
              ],
                [
                   23846,
                   23846,
                   23846
              ],
                [
                   8104,
                   8104,
                   8104
              ],
                [
                   10146,
                   10146,
                   10146
              ],
                [
                   14990,
                   14990,
                   14990
              ],
                [
                   51312,
                   51312,
                   51312
              ],
               [
                   3736,
                   3736,
                   3736
              ],
               [
                   76788,
                   76788,
                   76788
              ],
               [
                   64460,
                   64460,
                   64460
              ],
               [
                   FormatSigFigs(0.559626,2) + 0,
                   FormatSigFigs(0.000766401,2) + 0,
                   0
              ],
               [
                   FormatSigFigs(0.734631,2) + 0,
                   FormatSigFigs(0.00150969,2) + 0,
                   0
              ],
               [
                   FormatSigFigs(0.752221,2) + 0,
                   0,
                   0
              ],
               [
                   FormatSigFigs(0.832348,2) + 0,
                   0,
                   0
              ],
               [
                   FormatSigFigs(0.732889,2) + 0,
                   FormatSigFigs(0.00153436,2) + 0,
                   0
              ],
               [
                   FormatSigFigs(0.740529,2) + 0,
                   FormatSigFigs(0.0023971,2) + 0,
                   FormatSigFigs(1.94886e-05,2) + 0
              ],
               [
                   FormatSigFigs(0.755621,2) + 0,
                   FormatSigFigs(0.00267666,2) + 0,
                   0
              ],
               [
                   FormatSigFigs(0.731781,2) + 0,
                   FormatSigFigs(0.00171902,2) + 0,
                   0
              ],
               [
                   FormatSigFigs(0.732904,2) + 0 + 0,
                   FormatSigFigs(0.00139621,2) + 0,
                   FormatSigFigs(1.55135e-05,2) + 0
              ]
          ]
      ];

      return $data;
    } elsif($real_array) {
      return $db->dbReadOne('chr21',  44492170);
    } elsif($real_array_sigfig) {
      state $stuff = $db->dbReadOne('chr21',  44492170);

      sub set {
        my $val = shift;

        if(ref $val) {
          for my $v (@$val) {
            $v = set($v);
          }

          return $val;
        }

        if(looks_like_number($val)) {
          return int($val) == $val ? int($val) : FormatSigFigs($val, 2) + 0 ;
        }

        return $val;
      }

      for my $val (@$stuff) {
        $val = set($val);
      }

      my $cVal = encode_cbor($stuff);
      my $decoded = decode_cbor($cVal);

      my $mp = Data::MessagePack->new()->prefer_float32()->prefer_integer();
      my $mpVal = $mp->pack($stuff);
      my $decodedMp = $mp->unpack($mpVal);
      p $decoded;
      p $decodedMp;
      return $stuff;
    } elsif($small_hash) {
      return { a => 1, b => 2, c => 3, d => 'string', e => 5.018 };
    } else {    # "large data"
        $data_set_name = "large data structure";
        return [
            [ map { my $y= "$_"; $_ } 1 .. 10000 ], {@str}, {@str}, [ map { my $y= "$_"; $_ } 1 .. 10000 ],
            {@str}, [@rand], {@str}, {@str},
        ];
    }
}

if ($diagrams) {
    require SOOT;
    SOOT::Init(0);
    SOOT->import(":all");

    my ( $enc_data, $dec_data );
    $enc_data = cmpthese_to_sanity($encoder_result) if $encoder_result;
    $dec_data = cmpthese_to_sanity($decoder_result) if $decoder_result;

    foreach my $dia (
        [ "Encoder performance [1/s]", $enc_data ],
        [ "Decoder performance [1/s]", $dec_data ],
      )
    {
        my ( $title, $d ) = @$dia;
        next if not $d;
        $_->[0] =~ s/_/ /g, $_->[0] =~ s/sereal /sereal, / for @$d;
        make_bar_chart(
            substr( $title, 0, 3 ),
            $d,
            {
                title    => $title,
                filename => do {
                    my $x = $title;
                    $x =~ s/\[1\/s\]/per second/;
                    $data_set_name . " - " . $x;
                },
            }
        );
    }

    my %names = (
        "JSON::XS"                => 'json xs',
        "Data::Dumper::Limited"   => 'ddl',
        "Data::MessagePack"       => "msgpack",
        "Data::Dumper (1)"        => "dd noindent",
        "Data::Dumper (2)"        => "dd",
        "Storable"                => 'storable',
        "Sereal::Encoder"         => 'sereal',
        "Sereal::Encoder, Snappy" => 'sereal, snappy',
    );

    make_bar_chart(
        "size",
        [
            sort { $b->[1] <=> $a->[1] }
            map { $_->{size} ? [ $_->{name}, $_->{size} ] : () } values %meta
        ],
        {
            title    => "Encoded output sizes [bytes]",
            color    => kRed(),
            filename => $data_set_name . " - Encoded output sizes in bytes",
        },
    );
    SOOT->Run if not $diagram_output_dir;
}

sub make_bar_chart {
    my ( $name, $data, $opts ) = @_;
    my $h = TH1D->new( $name, ( $opts->{title} || $name ),
        scalar(@$data), -0.5, scalar(@$data) - 0.5 );
    $h->keep;
    $h->SetFillColor( $opts->{color} || kBlue() );
    $h->SetBarOffset(0.12);
    $h->SetBarWidth(0.74);
    $h->SetStats(0);
    $h->GetXaxis()->SetLabelSize(0.06);
    $h->GetXaxis()->SetLabelOffset(0.009);
    $h->GetYaxis()->SetTitle( $opts->{title} ) if defined $opts->{title};
    $h->GetYaxis()->SetTitleSize(0.045);

    for my $i ( 1 .. @$data ) {
        my ( $label, $rate ) = @{ $data->[ $i - 1 ] };
        $h->GetXaxis()->SetBinLabel( $i, $label );
        $h->SetBinContent( $i, 0 + $rate );
    }
    my $c = TCanvas->new->keep;
    $c->GetPad(0)->SetBottomMargin(0.175);
    $c->GetPad(0)->SetLeftMargin(0.15);
    $c->GetPad(0)->SetRightMargin(0.115);
    $c->GetPad(0)->SetGrid();
    $h->Draw("bar2");
    if ($diagram_output_dir) {
        require File::Path;
        File::Path::mkpath($diagram_output_dir);
        my $file = $opts->{filename}
          || do { my $f = $opts->{title}; $f =~ s/[^a-zA-Z0-9_\ ]/_/g; $f };
        $c->SaveAs("$diagram_output_dir/$file.png");
    }
}

sub cmpthese_to_sanity {
    my $res  = shift;
    my @rows = map {
        my $rate = $_->[1];
        if ( not $rate =~ s/\s*\/\s*s$// ) {
            $rate = 1 / $rate;
        }
        [ $_->[0], $rate ]
    } grep { defined $_->[0] and $_->[0] =~ /\S/ } @$res;
    return \@rows;
}
print "\n";
