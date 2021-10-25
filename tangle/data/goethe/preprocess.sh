#!/usr/bin/env bash

# extract data and convert to .json format

RAWTAG=""
if [[ $@ = *"--raw"* ]]; then
  RAWTAG="--raw"
fi
if [ ! -d "data/all_data" ] || [ ! "$(ls -A data/all_data)" ]; then
    cd preprocess
    ./data_to_json.sh $RAWTAG
    cd ..
fi

NAME="goethe"

cd ../utils

./preprocess.sh --name $NAME $@

cd ../$NAME
