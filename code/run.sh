#!/usr/bin/env bash

# Embeddings models
imdb="models/imdb.bin"
imdb_text8="models/imdb-text8.bin"
text8="models/text8.bin"
google="../word2vec-models/GoogleNews-vectors-negative300.bin.gz"


# run experiments

array=($imdb $imdb_text8 $text8 $google )

time

for vector in "${array[@]}"
do
	if [ ! -e ${vector} ]; then
		echo "$vector NOT FOUND! You get the vectors file."
		echo "	see 1-training-models.sh in models folder."
	else
		echo "%% running sentiment analysis test using $vector ... "
		echo
		time python sentiment.py -v ${vector} --details --output -log -td data/UMICH.csv
	fi
done
