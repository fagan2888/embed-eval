# By: A.Aziz Altowayan

import csv
import os
from logging import basicConfig, info, INFO

LOG_HEAD = '%(levelname)s: %(message)s'
basicConfig(format=LOG_HEAD, level=INFO)


def merge_reviews(pos_dir, neg_dir=None, type_=None):
    with open(type_ + '.csv', 'w') as fp:
        a = csv.writer(fp, delimiter='\t')
        header = 'sentiment', 'review'
        a.writerow(header)

        def add_review_rows(path, sentiment='NaN'):
            for f in os.listdir(path):
                if f.endswith('.txt'):
                    fpath = path + f
                    txt = open(fpath).read()
                    a.writerow([sentiment, txt])

        if type_ == 'imdb_unsup':
            add_review_rows(pos_dir)
        else:
            # positive
            add_review_rows(pos_dir, 1)
            # negative
            add_review_rows(neg_dir, 0)
        info('wrote {}.csv'.format(type_))


# merge the training review files
info('merging training review files ...')
train_pos = 'aclImdb/train/pos/'
train_neg = 'aclImdb/train/neg/'
# merge_reviews(train_pos, train_neg, 'imdb_train')

# merge the testing review files
info('merging testing review files ...')
train_pos = 'aclImdb/test/pos/'
train_neg = 'aclImdb/test/neg/'
# merge_reviews(train_pos, train_neg, 'imdb_test')

# merge the unsupervised review files
info('merging unsupervised review files ...')
train_unsup = 'aclImdb/train/unsup/'
# merge_reviews(train_unsup, type_='imdb_unsup')


# merge all reviews (text only) into one big file
def merge_all_imdb_reviews_into(fname):
    with open(fname, 'w') as ofile:
        def add_review_rows(path):
            for f in os.listdir(path):
                if f.endswith('.txt'):
                    fpath = path + f
                    txt = open(fpath).read()
                    ofile.write(txt)

        paths = [
            'aclImdb/train/pos/',
            'aclImdb/train/neg/',
            'aclImdb/test/pos/',
            'aclImdb/test/neg/',
            'aclImdb/train/unsup/'
        ]
        for path in paths:
            add_review_rows(path)


# big text file for all 100K reviews.
info('merging all review files into one big txt file ...')
outfile = 'imdb-100k.txt'
merge_all_imdb_reviews_into(outfile)

# cleaning big text
import re


def clean_txt(txt_file):
    txt = open(txt_file, 'r').read()
    print('cleaning ... ')
    # remove HTML
    txt = re.sub(r'<.*?>', '', txt)

    # remove non-letters
    txt = re.sub('[^a-zA-Z]', ' ', txt)

    # words to lower case and split them
    tokens = ' '.join(txt.lower().split())

    out_file = txt_file.replace('.txt', '-cleaned.txt')

    open(out_file, 'w').write(tokens)
    print('wrote cleaned file to {}'.format(out_file))


# clean the big file
info('cleaning the big txt file ...')
clean_txt(outfile)
