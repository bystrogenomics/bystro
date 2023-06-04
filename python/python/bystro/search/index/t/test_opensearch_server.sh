#!/usr/bin/env bash

# Starts a basic opensearch server, with synonyms mapped to the appropriate location for mapping to work
running=$(docker container list | grep "opensearchproject/opensearch:latest")

if [ -z "$running" ]
then
    docker pull opensearchproject/opensearch:latest

    docker run -d -p 9200:9200 -p 9600:9600 -e "discovery.type=single-node" -v "$PWD"/analysis:/usr/share/opensearch/config/analysis opensearchproject/opensearch:latest
else
  echo -e "\nOpensearch server is running: \n$running\n\n"
fi