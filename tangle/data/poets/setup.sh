#!/bin/bash
if [ ! -d "data" ]; then
    mkdir data
fi

if [ ! -d "data/test" ]; then
    mkdir data/test
    mkdir data/test-temp
fi

if [ ! -d "data/train" ]; then
    mkdir data/train
    mkdir data/train-temp
fi

# Copy goethe data and prefix it, so that it wont interfere with other files

cp ../goethe/data/test/* data/test-temp
cp ../goethe/data/train/* data/train-temp

cd data/test-temp
for f in * ; do mv -- "$f" "goethe_$f" ; done
cd ../train-temp
for f in * ; do mv -- "$f" "goethe_$f" ; done
cd ../..

mv data/train-temp/* data/train
mv data/test-temp/* data/test

# Copy shakespeare data and prefix it, so that it wont interfere with other files

cp ../shakespeare/data/test/* data/test-temp
cp ../shakespeare/data/train/* data/train-temp

cd data/test-temp
for f in * ; do mv -- "$f" "shakespeare_$f" ; done
cd ../train-temp
for f in * ; do mv -- "$f" "shakespeare_$f" ; done
cd ../..

mv data/train-temp/* data/train
mv data/test-temp/* data/test

rm -r data/*-temp
