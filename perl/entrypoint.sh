#!/bin/bash

# Determine which script to run based on the argument

case "$1" in
bystro-annotate.pl)
	exec bystro-annotate.pl "${@:2}" # Run bystro-annotate.pl and pass any additional arguments
	;;
bystro-build.pl)
	exec bystro-build.pl "${@:2}" # Run bystro-build.pl and pass any additional arguments
	;;
bystro-server.pl)
	exec bystro-server.pl "${@:2}" # Run bystro-server.pl and pass any additional arguments
	;;
bystro-utils.pl)
	exec bystro-utils.pl "${@:2}" # Run bystro-utils.pl and pass any additional arguments
	;;
read_db_util.pl)
	exec read_db_util.pl "${@:2}" # Run read_db_util.pl and pass any additional arguments
	;;
*)
	echo "Usage: docker run bystro {bystro-annotate.pl|bystro-build.pl|bystro-server.pl|bystro-utils.pl|read_db_util.pl} [args]"
	exit 1
	;;
esac
