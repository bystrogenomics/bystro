#!/usr/bin/env bash

echo "Installing LMDB"

# Bystro uses LMDB as its db engine. Fast, great use of cache
rm -rf lmdb
git clone git://github.com/LMDB/lmdb.git;
make -C lmdb/libraries/liblmdb;
sudo make install -C lmdb/libraries/liblmdb;
rm -rf lmdb