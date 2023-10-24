#Run from ../lib
use 5.10.0;
use strict;
use warnings;

package Mock;

use Mouse 2;

with 'Seq::Role::Message', 'Seq::Role::IO';

has logPath =>
  ( is => 'ro', init_arg => undef, default => 't/role/02-message.test.log' );

sub BUILD {
  my $self = shift;

  $self->setLogPath( $self->logPath );
}

1;

package TestMessage;
use Test::More;
use DDP;

my $mocker = Mock->new();

$mocker->log( 'warn', "HELLO WORLD" );

my ( $err, undef, $fh ) = $mocker->getReadFh( $mocker->logPath );
ok( !$err, 'No error gets generated on reading fh' );

my @lines = <$fh>;

$err = $mocker->safeClose($fh);
ok( !$err, 'No error gets generated on closing fh' );

ok( @lines == 1,                            'Only one line gets written' );
ok( index( $lines[0], "HELLO WORLD" ) > -1, 'By default warnings allowed' );

$mocker = Mock->new();

( $err, undef, $fh ) = $mocker->getReadFh( $mocker->logPath );
ok( !$err, 'No error gets generated on reading fh nth time' );

@lines = <$fh>;

$err = $mocker->safeClose($fh);
ok( !$err, 'No error gets generated on closing fh nth time' );

ok( @lines == 0, "Log file gets cleared" );

$mocker->setLogLevel('fatal');

$mocker->log( 'warn', "A different warning" );

( $err, undef, $fh ) = $mocker->getReadFh( $mocker->logPath );
ok( !$err, 'No error gets generated on reading fh nth time' );

@lines = <$fh>;

$err = $mocker->safeClose($fh);
ok( !$err, 'No error gets generated on closing fh nth time' );

ok( !@lines,
  "setLogLevel sets log level to fatal, and warning messages don't get stored" );

$mocker = Mock->new();

$mocker->setLogLevel('info');

$mocker->log( 'warn', "A warning above info" );

( $err, undef, $fh ) = $mocker->getReadFh( $mocker->logPath );
ok( !$err, 'No error gets generated on reading fh nth time' );

@lines = <$fh>;
$err   = $mocker->safeClose($fh);

ok( !$err, 'No error gets generated on closing fh nth time' );

ok(
  index( $lines[0], 'A warning above info' ) > -1,
  "Role::Message sets info level, and writes warning messages"
);

$mocker = Mock->new();
$mocker->log( 'warn',  "A new warning above info" );
$mocker->log( 'info',  "An info message" );
$mocker->log( 'error', "An error message" );

( $err, undef, $fh ) = $mocker->getReadFh( $mocker->logPath );
ok( !$err, 'No error gets generated on reading fh nth time' );

@lines = <$fh>;

$err = $mocker->safeClose($fh);
ok( !$err, 'No error gets generated on closing fh nth time' );

ok( @lines == 3, 'Role::Message properly writes multiple lines' );
ok(
  index( $lines[0], 'A new warning above info' ) > -1,
  "Role::Message doesn't overwrite previous messages"
);
ok(
  index( $lines[1], 'An info message' ) > -1,
  "Role::Message records info messages at info level"
);
ok(
  index( $lines[1], 'An info message' ) > -1,
  "Role::Message records info messages at info level"
);

use Try::Tiny;

try {
  $mocker->log( 'fatal', "A fatal message" );
}
catch {
  ok( $_ && index( $_, 'A fatal message' ) > -1,
    "Role::Message throws a fatal message at info level: $_" );
};

system( 'rm ' . $mocker->logPath );

done_testing();
