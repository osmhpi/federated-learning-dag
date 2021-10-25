#!/usr/bin/env bash

# This was the original way to get the data, but it is blocked now in Germany

# cd ./data/raw_data
# wget http://www.gutenberg.org/files/100/old/1994-01-100.zip
# unzip 1994-01-100.zip
# rm 1994-01-100.zip
# mv 100.txt raw_data.txt
# cd ../../preprocess

# We added the zip file into the git repo, so we still can generate data

unzip ../1994-01-100.zip -d ./data/raw_data
mv ./data/raw_data/100.txt ./data/raw_data/raw_data.txt
