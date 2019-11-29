#!/bin/bash

if [ ! -e word2vec/compute-accuracy ]; then
	echo "******************************"
	echo "Please download word2vec first."
	echo "******************************"
	exit 1
fi

# Embeddings models
imdb="imdb.bin"
imdb_text8="imdb-text8.bin"
text8="text8.bin"
google="../../word2vec-models/GoogleNews-vectors-negative300.bin"

MODELS=( $imdb $imdb_text8 $text8 $google )

for MODEL in "${MODELS[@]}"
do
	if [ ! -e $MODEL ]; then
		echo "$MODEL not found."
	else
		echo "---- Evaluating the accuracy of $MODEL -----"
		time ./word2vec/compute-accuracy $MODEL 30000 < word2vec/questions-words.txt
	fi
done

# to compute accuracy with the full vocabulary, use:
# ./word2vec/compute-accuracy vectors.bin < word2vec/questions-words.txt
# NOTE: It will take way longer time.
