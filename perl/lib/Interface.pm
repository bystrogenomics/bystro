#!/usr/bin/env perl
use 5.10.0;

package Interface;

use File::Basename;

use Mouse;

use Path::Tiny;
use Mouse::Util::TypeConstraints;

use namespace::autoclean;

use YAML::XS qw/LoadFile/;
use JSON::XS;

use Getopt::Long::Descriptive;

use Try::Tiny;

use Seq;
with 'MouseX::Getopt';

##########Parameters accepted from command line#################
has input => (
  is            => 'ro',
  isa           => 'ArrayRef[Str]',
  metaclass     => 'Getopt',
  cmd_aliases   => [qw/i in/],
  documentation =>
    'Input files. Supports mulitiple files: --in file1 --in file2 --in file3',
);

has output => (
  is            => 'ro',
  isa           => 'Str',
  cmd_aliases   => [qw/o out/],
  metaclass     => 'Getopt',
  documentation => 'Base path for output files: /path/to/output',
);

has json => (
  is            => 'ro',
  isa           => 'Bool',
  metaclass     => 'Getopt',
  documentation =>
    'Do you want to output JSON instead? Incompatible with run_statistics',
);

has config => (
  is            => 'ro',
  isa           => 'Str',
  coerce        => 1,
  required      => 0,
  metaclass     => 'Getopt',
  cmd_aliases   => [qw/c configuration/],
  documentation => 'Yaml config file path.',
);

has overwrite => (
  is            => 'ro',
  isa           => 'Int',
  default       => 0,
  required      => 0,
  metaclass     => 'Getopt',
  documentation => 'Overwrite existing output file.',
);

has read_ahead => (
  is            => 'ro',
  isa           => 'Bool',
  default       => 0,
  coerce        => 1,
  required      => 0,
  metaclass     => 'Getopt',
  documentation => 'For dense datasets, use system read-ahead',
);

has debug => (
  is        => 'ro',
  isa       => 'Num',
  default   => 0,
  required  => 0,
  metaclass => 'Getopt',
);

has verbose => (
  is        => 'ro',
  isa       => 'Int',
  required  => 0,
  metaclass => 'Getopt',
);

has compress => (
  is            => 'ro',
  isa           => 'Str',
  metaclass     => 'Getopt',
  documentation => 'Compress the output?',
  default       => 0,
);

has archive => (
  is            => 'ro',
  isa           => 'Bool',
  metaclass     => 'Getopt',
  documentation => 'Place all outputs into a tarball?',
  default       => 0,
);

has run_statistics => (
  is            => 'ro',
  isa           => 'Int',
  metaclass     => 'Getopt',
  documentation =>
    'Create per-sample feature statistics (like transition:transversions)?',
  default => 1,
);

has delete_temp => (
  is            => 'ro',
  isa           => 'Int',
  documentation => 'Delete the temporary directory made during annotation',
  default       => 1,
);

has wantedChr => (
  is            => 'ro',
  isa           => 'Str',
  metaclass     => 'Getopt',
  cmd_aliases   => [qw/chr wanted_chr/],
  documentation => 'Annotate a single chromosome',
);

has maxThreads => (
  is            => 'ro',
  isa           => 'Int',
  metaclass     => 'Getopt',
  cmd_aliases   => [qw/threads/],
  documentation => 'Number of CPU threads to use (optional)',
);

has publisher => (
  is            => 'ro',
  isa           => 'Str',
  required      => 0,
  metaclass     => 'Getopt',
  documentation =>
    'Tell Bystro how to send messages to a plugged-in interface (such as a web interface)'
);

has ignore_unknown_chr => (
  is            => 'ro',
  isa           => 'Bool',
  default       => 1,
  required      => 0,
  metaclass     => 'Getopt',
  documentation => 'Don\'t quit if we find a non-reference chromosome (like ChrUn)'
);

has json_config => (
  is            => 'ro',
  isa           => 'Str',
  required      => 0,
  metaclass     => 'Getopt',
  documentation =>
    'JSON config file path. Use this if you wish to invoke the annotator by file passing.',
);

has result_summary_path => (
  is            => 'ro',
  isa           => 'Str',
  required      => 0,
  metaclass     => 'Getopt',
  documentation => 'Where to output the result summary. Defaults to STDOUT',
);

sub annotate {
  my $self = shift;

  my $args;

  if ( $self->json_config ) {
    my $json_config_data = path( $self->json_config )->slurp;
    # p $json_config_data;
    $args = decode_json($json_config_data);
  }
  else {
    my $publisher;

    if ( $self->publisher ) {
      if ( type $self->publisher eq 'Str' ) {
        $publisher = decode_json( $self->publisher );
      }
      else {
        $publisher = $self->publisher;
      }
    }

    if ( defined $self->verbose ) {
      $args->{verbose} = $self->verbose;
    }

    if ( defined $self->maxThreads ) {
      $args->{maxThreads} = $self->maxThreads;
    }

    if ( defined $self->json ) {
      $args->{outputJson} = $self->json;

      if ( $self->run_statistics ) {
        say STDERR "--json incompatible with --run_statistics 1";
        exit(1);
      }
    }

    $args = {
      config             => $self->config,
      input_files        => $self->input,
      output_file_base   => $self->output,
      debug              => $self->debug,
      wantedChr          => $self->wantedChr,
      ignore_unknown_chr => $self->ignore_unknown_chr,
      overwrite          => $self->overwrite,
      publisher          => $publisher,
      compress           => $self->compress,
      archive            => $self->archive,
      run_statistics     => !!$self->run_statistics,
      delete_temp        => !!$self->delete_temp,
      readAhead          => $self->read_ahead
    };
  }

  my ( $err, $results, $totalProgress, $totalSkipped );

  try {
    my $annotator = Seq->new_with_config($args);
    ( $err, $results, $totalProgress, $totalSkipped ) = $annotator->annotate();
  }
  catch {
    $err = $_;
  };

  my $formattedResults = JSON::XS->new->pretty(1)->encode(
    {
      error         => $err,
      results       => $results,
      totalProgress => $totalProgress,
      totalSkipped  => $totalSkipped,
    }
  );

  if ( $self->result_summary_path ) {
    path( $self->result_summary_path )->spew($formattedResults);
  }
  else {
    say $formattedResults;
  }
}

__PACKAGE__->meta->make_immutable;

1;

=item messanger

Contains a hash reference (also accept json representation of hash) that
tells Bystro how to send data to a plugged interface.

Example: {
      room: jobObj.userID,
      message: {
        publicID: jobObj.publicID,
        data: tData,
      },
    };
=cut
