#!/bin/bash

#### Training shakespeare text ####

cat ../data/corpus/imdb-text8.txt ../data/corpus/shakespeare-clean.txt > ../data/corpus/db-t8-shk.txt

CORPUS="../data/corpus/db-t8-shk.txt"
OUT="db-t8-shk.bin"

echo "----- generating vector embeddings for $CORPUS -----"
time ./word2vec/word2vec -train $CORPUS -output $OUT -cbow 1 -size 300 -window 10 -negative 25 -hs 0 -sample 1e-4 -threads 20 -binary 1 -iter 15
