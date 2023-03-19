#!/usr/bin/env bash

## TODO: add beanstalkd local server and trigger submit of job
. ./test_opensearch_server.sh
. ../../../../.initialize_conda_env.sh
python ../listener.py --conf_dir ./ --queue_conf  --search_conf ./test_hg19.mapping.yml --queue_conf ./beanstalk.yml