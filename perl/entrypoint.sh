#!/bin/bash

# Determine which script to run based on the argument

case "$1" in
bystro-annotate.pl)
	exec /usr/local/bin/perl /app/bin/bystro-annotate.pl "${@:2}" # Run bystro-annotate.pl and pass any additional arguments
	;;
bystro-build.pl)
	exec /usr/local/bin/perl /app/bin/bystro-build.pl "${@:2}" # Run bystro-build.pl and pass any additional arguments
	;;
bystro-server.pl)
	exec /usr/local/bin/perl /app/bin/bystro-server.pl "${@:2}" # Run bystro-server.pl and pass any additional arguments
	;;
bystro-utils.pl)
	exec /usr/local/bin/perl /app/bin/bystro-utils.pl "${@:2}" # Run bystro-utils.pl and pass any additional arguments
	;;
read_db_util.pl)
	exec /usr/local/bin/perl /app/bin/read_db_util.pl "${@:2}" # Run read_db_util.pl and pass any additional arguments
	;;
*)
	echo "Usage: docker run bystro {bystro-annotate.pl|bystro-build.pl|bystro-server.pl|bystro-utils.pl|read_db_util.pl} [args]"
	exit 1
	;;
esac
