#!/usr/bin/env bash

NAME="goethe"

cd ../utils

python3 stats.py --name $NAME

cd ../$NAME
