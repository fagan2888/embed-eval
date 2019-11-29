#!/bin/bash

if [ ! -e word2vec/word2vec ]; then
	echo "******************************"
	echo "Please download word2vec first."
	echo "******************************"
	exit 1
fi


#### Training IMDB corpus ####

CORPUS="../data/corpus/imdb-100k-cleaned.txt"
OUT="imdb.bin"

if [ ! -e $CORPUS ]; then
	echo "*******************************"
	echo "please download the data first."
	echo "See: ../data/download-data/"
	echo "*******************************"
	exit 1
fi
echo "----- generating vector embeddings for $CORPUS -----"
time ./word2vec/word2vec -train $CORPUS -output $OUT -cbow 1 -size 300 -window 10 -negative 25 -hs 0 -sample 1e-4 -threads 20 -binary 1 -iter 15


#### Training IDMB-text8 corpus ####

CORPUS="../data/corpus/imdb-text8.txt"
OUT="imdb-text8.bin"

if [ ! -e $CORPUS ]; then
	if [ ! -e ../data/corpus/text8 ]; then
		wget http://mattmahoney.net/dc/text8.zip -P ../data/corpus/
		echo "Uncompressing text8 corpus ..."
		time tar -zxf ../data/corpus/text8.zip -C ../data/corpus/
	fi
	cat ../data/corpus/imdb-100k-cleaned.txt ../data/corpus/text8 > ../data/corpus/imdb-text8.txt
fi
echo "----- generating vector embeddings for $CORPUS -----"
time ./word2vec/word2vec -train $CORPUS -output $OUT -cbow 1 -size 300 -window 10 -negative 25 -hs 0 -sample 1e-4 -threads 20 -binary 1 -iter 15



#### Training text8 corpus ####

CORPUS="../data/corpus/text8"
OUT="text8.bin"

echo "----- generating vector embeddings for $CORPUS -----"
time ./word2vec/word2vec -train $CORPUS -output $OUT -cbow 1 -size 300 -window 10 -negative 25 -hs 0 -sample 1e-4 -threads 20 -binary 1 -iter 15


#### GoogleNews pre-trained model
echo
echo "GoogleNews-vectors-negative300.bin.gz is a pre-trained model."
echo "See: https://code.google.com/archive/p/word2vec/ to download it (file size 1.65 GB)."
echo
