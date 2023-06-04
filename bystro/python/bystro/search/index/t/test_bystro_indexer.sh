#!/usr/bin/env bash

## TODO: add beanstalkd local server and trigger submit of job
. ./test_opensearch_server.sh
. ../../../../.initialize_conda_env.sh
python ../handler.py --tar ./13073_2016_396_moesm2_esm-2.tar --index_name "test_13073_2016_396_moesm2_esm" --search_conf ./test_opensearch_config.yml --mapping_conf ./test_hg19.mapping.yml --annotation_conf ./hg19.yml 