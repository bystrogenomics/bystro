#!/usr/bin/env perl

use strict;
use warnings;
use Test::More tests => 23;
use Test::MockModule;
use Cpanel::JSON::XS;
use FindBin;
use lib "$FindBin::Bin/../";

require 'bystro-server.pl';

# Mock Beanstalk::Client
my $mock_beanstalk = Test::MockModule->new('Beanstalk::Client');

$mock_beanstalk->mock(
  'new',
  sub {
    return bless {
      server          => $_[1]->{server},
      default_tube    => $_[1]->{default_tube},
      connect_timeout => $_[1]->{connect_timeout},
      encoder         => $_[1]->{encoder} // \&YAML::Syck::Dump,
      decoder         => $_[1]->{decoder} // \&YAML::Syck::Load,
      socket          => undef,
      error           => undef,
      },
      'Beanstalk::Client';
  }
);

$mock_beanstalk->mock(
  'put',
  sub {
    my ( $self, $opt ) = @_;
    return bless {
      id     => 1,
      client => $self,
      data   => $opt->{data},
      },
      'Beanstalk::Job';
  }
);

$mock_beanstalk->mock(
  'reserve',
  sub {
    my ( $self, $timeout ) = @_;
    return bless {
      id     => 1,
      client => $self,
      data   => encode_json( { submissionId => 'test_id' } ),
      },
      'Beanstalk::Job';
  }
);

$mock_beanstalk->mock(
  'delete',
  sub {
    return 1;
  }
);

$mock_beanstalk->mock(
  'release',
  sub {
    return 1;
  }
);

$mock_beanstalk->mock(
  'error',
  sub {
    return undef;
  }
);

# Mock job object
my $mock_job = Test::MockModule->new('Beanstalk::Job');

$mock_job->mock(
  'id',
  sub {
    return 1;
  }
);

$mock_job->mock(
  'data',
  sub {
    return encode_json( { submissionId => 'test_id' } );
  }
);

$mock_job->mock(
  'delete',
  sub {
    return 1;
  }
);

$mock_job->mock(
  'release',
  sub {
    return 1;
  }
);

$mock_job->mock(
  'stats',
  sub {
    return { id => 1, state => 'reserved' };
  }
);

# Test the main functions
# use beanstalkd_server qw(_getConsumerClient _getProducerClient putWithTimeout reserveWithTimeout deleteWithTimeout releaseWithTimeout statsWithTimeout);
my $producer = _getProducerClient( 'localhost', 'test_tube' );
isa_ok( $producer, 'Beanstalk::Client', 'Producer client created' );

my $consumer = _getConsumerClient( 'localhost', 'test_tube' );
isa_ok( $consumer, 'Beanstalk::Client', 'Consumer client created' );

my $err = putWithTimeout( $producer, 5,
  { priority => 0, data => encode_json( { test => 'data' } ) } );
is( $err, undef, 'putWithTimeout successful' );

my ( $reserve_err, $job ) = reserveWithTimeout( $consumer, 5 );
is( $reserve_err, undef, 'reserveWithTimeout successful' );
isa_ok( $job, 'Beanstalk::Job', 'Job reserved' );

$err = deleteWithTimeout( $job, 5 );
is( $err, undef, 'deleteWithTimeout successful' );

$err = releaseWithTimeout( $job, 5 );
is( $err, undef, 'releaseWithTimeout successful' );

my ( $stats_err, $stats ) = statsWithTimeout( $job, 5 );
is( $stats_err, undef, 'statsWithTimeout successful' );
is_deeply( $stats, { id => 1, state => 'reserved' }, 'Job stats correct' );

# Mock Beanstalk::Client
$mock_beanstalk = Test::MockModule->new('Beanstalk::Client');

$mock_beanstalk->mock(
  'new',
  sub {
    return bless {
      server          => $_[1]->{server},
      default_tube    => $_[1]->{default_tube},
      connect_timeout => $_[1]->{connect_timeout},
      encoder         => $_[1]->{encoder} // \&YAML::Syck::Dump,
      decoder         => $_[1]->{decoder} // \&YAML::Syck::Load,
      socket          => undef,
      error           => undef,
      },
      'Beanstalk::Client';
  }
);

$mock_beanstalk->mock(
  'put',
  sub {
    my ( $self, $opt ) = @_;
    if ( $self->{force_error} ) {
      $self->{error} = "forced error";
      return undef;
    }
    if ( $self->{hang} ) {
      sleep(2);
    }
    return bless {
      id     => 1,
      client => $self,
      data   => $opt->{data},
      },
      'Beanstalk::Job';
  }
);

$mock_beanstalk->mock(
  'reserve',
  sub {
    my ( $self, $timeout ) = @_;
    if ( $self->{force_error} ) {
      $self->{error} = "forced error";
      return undef;
    }
    if ( $self->{hang} ) {
      sleep(2);
    }
    return bless {
      id     => 1,
      client => $self,
      data   => encode_json( { submissionId => 'test_id' } ),
      },
      'Beanstalk::Job';
  }
);

$mock_beanstalk->mock(
  'delete',
  sub {
    my ( $self, $id ) = @_;
    if ( $self->{force_error} ) {
      $self->{error} = "forced error";
      return undef;
    }
    if ( $self->{hang} ) {
      sleep(2);
    }
    return 1;
  }
);

$mock_beanstalk->mock(
  'release',
  sub {
    my ( $self, $id ) = @_;
    if ( $self->{force_error} ) {
      $self->{error} = "forced error";
      return undef;
    }
    if ( $self->{hang} ) {
      sleep(2);
    }
    return 1;
  }
);

$mock_beanstalk->mock(
  'error',
  sub {
    my $self = shift;
    return $self->{error};
  }
);

# Mock job object
# Mock Beanstalk::Job
my $mock_job = Test::MockModule->new('Beanstalk::Job');

$mock_job->mock(
  'new',
  sub {
    return bless {
      id       => $_[1]->{id},
      client   => $_[1]->{client},
      data     => $_[1]->{data},
      reserved => $_[1]->{reserved} // 0,
      buried   => $_[1]->{buried}   // 0,
      error    => undef,
      },
      'Beanstalk::Job';
  }
);

$mock_job->mock(
  'stats',
  sub {
    my $self = shift;
    if ( $self->client->{force_error} ) {
      $self->{error} = "forced error";
      return undef;
    }
    return { id => $self->id, state => 'reserved' };
  }
);

$mock_job->mock(
  'delete',
  sub {
    my $self = shift;
    my $ret  = $self->client->delete( $self->id );
    unless ($ret) {
      $self->error( $self->client->error );
      return undef;
    }
    $self->reserved(0);
    $self->buried(0);
    return 1;
  }
);

$mock_job->mock(
  'touch',
  sub {
    my $self = shift;
    my $ret  = $self->client->touch( $self->id );
    unless ($ret) {
      $self->error( $self->client->error );
      return undef;
    }
    return 1;
  }
);

$mock_job->mock(
  'release',
  sub {
    my ( $self, $opt ) = @_;
    my $ret = $self->client->release( $self->id, $opt );
    unless ($ret) {
      $self->error( $self->client->error );
      return undef;
    }
    $self->reserved(0);
    return 1;
  }
);

$mock_job->mock(
  'error',
  sub {
    my $self = shift;
    return $self->{error};
  }
);

local $beanstalkd_server::BEANSTALKD_RESERVE_TIMEOUT = 1;

$producer = _getProducerClient( 'localhost', 'test_tube' );
isa_ok( $producer, 'Beanstalk::Client', 'Producer client created' );

$consumer = _getConsumerClient( 'localhost', 'test_tube' );
isa_ok( $consumer, 'Beanstalk::Client', 'Consumer client created' );

# Successful putWithTimeout
$err = putWithTimeout( $producer, 5,
  { priority => 0, data => encode_json( { test => 'data' } ) } );
is( $err, undef, 'putWithTimeout successful' );

# Simulate error in putWithTimeout
$producer->{force_error} = 1;
$err = putWithTimeout( $producer, 5,
  { priority => 0, data => encode_json( { test => 'data' } ) } );
is( $err, 'forced error', 'putWithTimeout with forced error' );

# Simulate hang in putWithTimeout
$producer->{force_error} = 0;
$producer->{hang}        = 1;
$err                     = putWithTimeout( $producer, 1,
  { priority => 0, data => encode_json( { test => 'data' } ) } );
like( $err, qr/TIMED_OUT_CLIENT_SIDE/, 'putWithTimeout with client-side timeout' );

# Successful reserveWithTimeout
$consumer->{hang} = 0;
( $reserve_err, $job ) = reserveWithTimeout( $consumer, 5 );
is( $reserve_err, undef, 'reserveWithTimeout successful' );
isa_ok( $job, 'Beanstalk::Job', 'Job reserved' );

# Successful deleteWithTimeout
$err = deleteWithTimeout( $job, 5 );
is( $err, undef, 'deleteWithTimeout successful' );

# Simulate error in deleteWithTimeout
$consumer->{force_error} = 1;
$err = deleteWithTimeout( $job, 5 );
is( $err, 'forced error', 'deleteWithTimeout with forced error' );

# Simulate hang in deleteWithTimeout
$consumer->{force_error} = 0;
$consumer->{hang}        = 1;
$err                     = deleteWithTimeout( $job, 1 );
like( $err, qr/TIMED_OUT_CLIENT_SIDE/,
  'deleteWithTimeout with client-side timeout' );

# Simulate error in reserveWithTimeout
$consumer->{force_error} = 1;
( $reserve_err, $job ) = reserveWithTimeout( $consumer, 5 );
is( $reserve_err, 'forced error', 'reserveWithTimeout with forced error' );
is( $job,         undef,          'No job reserved due to error' );

# Simulate hang in reserveWithTimeout
$consumer->{force_error} = 0;
$consumer->{hang}        = 1;
( $reserve_err, $job ) = reserveWithTimeout( $consumer, 1 );
like( $reserve_err, qr/TIMED_OUT_CLIENT_SIDE/,
  'reserveWithTimeout with client-side timeout' );
is( $job, undef, 'No job reserved due to timeout' );

done_testing();
