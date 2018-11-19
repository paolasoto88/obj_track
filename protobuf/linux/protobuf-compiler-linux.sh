#!/usr/bin/env bash

sudo apt-get install protobuf-compiler
cd ../../models/research
protoc object_detection/protos/*.proto --python_out=.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim