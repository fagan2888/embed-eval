# By: A.Aziz Altowayan


## standard libs
import logging
import argparse
from logging import info
import os  # for writing results to files
import re
## dependency libs
import pandas as pd
from gensim.models import Word2Vec
import numpy as np
# classification and plotting
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegressionCV, Perceptron
from sklearn.svm import LinearSVC, NuSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import roc_curve, auc

# logging parameters
param = {
    'format': '[%(asctime)s] %(levelname)s: %(message)s',
    'datefmt': '%H:%M:%S',
    'level': logging.INFO,
}


# read data
def read_data(train_in, test_in):
    train_df = pd.read_csv(train_in, delimiter="\t")
    test_df = pd.read_csv(test_in, delimiter="\t")
    info('loaded {} {} and {} {}'.format(train_in, train_df.shape, test_in, test_df.shape))

    return train_df, test_df


# dataset preprocessing and cleaning.
def tokenize(text):
    """Input: a sentence (review)
       Return: a list of tokens"""

    # remove HTML
    text = re.sub(r'<.*?>', '', text)

    # remove non-letters
    text = re.sub('[^a-zA-Z]', ' ', text)

    # words to lower case and split them
    words = text.lower().split()

    return words, len(words)


def tokenize_data(reviews_text, type_='NaN'):
    """ Input: a list of sentences (reviews)
        Output: a list of tokens for each sentence."""

    txt_tokens = []
    info('Tokenizing the {} reviews text ... '.format(type_))
    total_tokens = []
    for review in reviews_text:
        tokens, num = tokenize(review)
        txt_tokens.append(tokens)
        total_tokens.append(num)
    tsum = sum(total_tokens)
    info('total {} {} tokens. Average {} per review.'.format(tsum, type_, tsum / len(reviews_text)))
    return txt_tokens


# #### loading the pre-trained word2vec model
def load_model(model_file_name):
    w2v_model = Word2Vec.load_word2vec_format(model_file_name, binary=True)
    # info('loaded {}'.format(model_name))
    w2v_model.init_sims(replace=True)  # to save memory
    vocab, vector_dim = w2v_model.syn0.shape
    # info('The model shape: {} {} (Vocabulary, dimension)'.format(vocab, vector_dim))
    return w2v_model, vector_dim


# Use the model to query the vector representation of each word in the review.
# Then average the retrieved vectors (into one vector) to be used as the review's feature.
# Note that any word that is not available in the model's vocabulary will be ignored.


def review_feature(words, vectors, dim=300):
    # average the review tokens vectors

    feature_vec = np.zeros((dim,), dtype="float32")
    review_len = len(words)
    retrieved_words = 0
    # missed_words = 0
    for token in words:
        try:
            feature_vec = np.add(feature_vec, vectors[token])
            retrieved_words += 1
        except KeyError:
            pass
            # missed_words += 1   # if word not in model

    feature_vec = np.divide(feature_vec, retrieved_words)

    # record missed words
    retrieval.append((review_len, retrieved_words))

    return feature_vec


def average_feature_vectors(reviews, model, dim, type_='NaN'):
    # Input: a list of lists (each list contains the review tokens)
    # e.g. [['hi','do'], ['you','see'],...]
    # for every list,
    #   get the corresponding vectors of its tokens from the embeddings model, then
    #   calculate its average vector

    global retrieval
    retrieval = []

    feature_vectors = np.zeros((len(reviews), dim), dtype="float32")
    logging.info("Retrieving vectors of {} tokens ...".format(type_))
    counter = 0
    for review in reviews:
        if counter % 20000 == 0:
            info("review {} of {}".format(counter, len(reviews)))

        # The feature of a review is the average of its tokens' embeddings
        feature_vectors[counter] = review_feature(review, model, dim)

        counter += 1

    return feature_vectors, retrieval


def plot_auc(fpr, tpr, roc_auc, title, classifier):
    plt.figure(figsize=(8, 4))
    plt.plot(fpr, tpr, label='ROC = {:.2f}'.format(roc_auc))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.05, 1.0]);
    plt.ylim([0.0, 1.05])
    plt.legend(loc='lower right')
    plt.title(title)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid()
    if write_results:
        if not os.path.exists('figures'):
            os.mkdir('figures')
        plt.savefig('figures/{}.png'.format(model_name + classifier), bbox_inches='tight')
    else:
        plt.show()


def classify(classifier=LogisticRegressionCV(), info_=False, plot=False):
    classifier_name = classifier.__class__.__name__
    if info_:
        info('fitting data ...')
        info('\n\ncreated \n\n{}'.format(classifier))

    classifier.fit(X_train, y_train)
    score = classifier.score(X_test, y_test) * 100
    pscore = '{:.2f}%'.format(score)
    if info_:
        info('\n\n\t{}() ACCURACY: {}\n'.format(classifier_name, pscore))
    else:
        info('\t{} accuracy: {}'.format(classifier_name, pscore))

    # prediction
    def total_predicted(class_):
        return len(classifier.predict(X_test)[classifier.predict(X_test) == class_])

    negative = total_predicted(0)
    positive = total_predicted(1)

    accuracies[classifier_name] = [pscore, positive, negative]

    # roc
    if classifier_name in ['Perceptron', 'LinearSVC']:
        classifier_probas = classifier.decision_function(X_test)
    else:
        classifier_probas = classifier.predict_proba(X_test)[:, 1]

    false_positive, true_positive, thresholds = roc_curve(y_test, classifier_probas)
    roc_auc = auc(false_positive, true_positive)

    # plot
    if plot:
        title = '{}(), accuracy: {:.2f}%, negative: {}, positive: {}'.format(classifier_name, score, negative, positive)
        plot_auc(false_positive, true_positive, roc_auc, title, classifier_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vectors",
                        help="Model contains vector embeddings of words (default: models/imdb.bin)")
    parser.add_argument("-t", "--train_dataset",
                        help="Reviews dataset to use for training the sentiment classifiers. "
                             "(default: data/imdb_train.csv)")
    parser.add_argument("-td", "--test_dataset",
                        help="Reviews dataset to use for testing the sentiment classifiers. "
                             "(default: data/test_train.csv)")
    parser.add_argument("-p", "--plot", help="To plot classifiers score result and ROC. (default: False)",
                        action="store_true")
    parser.add_argument("-d", "--details", help="To output classifiers parameters info.(default: False)",
                        action="store_true")
    parser.add_argument("-o", "--output", help="To write the result and figures to external file(s). (default: False)",
                        action="store_true")
    parser.add_argument("-log", "--logging", help="To write logging info to an external file. (default: False)",
                        action="store_true")
    args = parser.parse_args()

    ## word embeddings model
    if args.vectors:
        model_file = args.vectors
    else:
        model_file = "models/imdb.bin"

    ## datasets
    if args.train_dataset:
        train_file = args.train_data
    else:
        train_file = "data/imdb_train.csv"

    if args.test_dataset:
        test_file = args.test_dataset
    else:
        test_file = "data/imdb_test.csv"

    ## files output
    global write_results
    if args.output:
        write_results = True
    else:
        write_results = False

    ## logging output
    if args.logging:
        out_log = True
    else:
        out_log = False

    if out_log:
        param['filename'] = 'sentiment.log'
        param['filemode'] = 'a'
    logging.basicConfig(**param)

    global model_name
    model_name = model_file.replace('.bin', '').split('/')[-1]

    # start here
    info('word2vec model ...')
    model, vector_dimension = load_model(model_file)

    train, test = read_data(train_file, test_file)

    train_review, test_review = train['review'], test['review']
    train_tokens = tokenize_data(train_review, 'training')
    test_tokens = tokenize_data(test_review, 'testing')

    # average retrieval error
    def avg_retrieval_err(ratio, num_queries):
        """ratio is a list of tuple [ (num_query_tokens, num_retrieved), ..]"""
        # report avg_rr
        differences = [t - r for t, r in ratio]
        avg_err = float(sum(differences)) / num_queries
        info('Average retrieval ERROR {} per review'.format(avg_err))

    def write_df(retrieve_ratio, type_):
        """write model queries stat result to a dataframe"""
        df = pd.DataFrame(retrieve_ratio, columns=['num_review_tokens', 'retrieved_vectors'])
        df['missed'] = df.num_review_tokens - df.retrieved_vectors
        if not os.path.exists('results'):
            os.mkdir('results')
        df.to_csv('results/ratio_{}-{}.csv'.format(model_name, type_))

    # averaging training tokens
    train_vectors, retrieval_ratio = average_feature_vectors(train_tokens, model, vector_dimension, 'training')
    # report missed words of training queries
    avg_retrieval_err(retrieval_ratio, len(train_vectors))
    if write_results:
        write_df(retrieval_ratio, 'train')

    # average testing reviews
    test_vectors, retrieval_ratio = average_feature_vectors(test_tokens, model, vector_dimension, 'testing')
    # report missing words of test queries
    avg_retrieval_err(retrieval_ratio, len(test_vectors))
    if write_results:
        write_df(retrieval_ratio, 'test')

    # ## 3. Sentiment classification of the reviews
    info('Sentiment CLASSIFIERS ... \n')

    global X_train, y_train, X_test, y_test
    X_train = train_vectors
    y_train = train['sentiment']
    X_test = test_vectors
    y_test = test['sentiment']

    # init binary classifiers
    classifiers = [
        RandomForestClassifier(n_estimators=100),
        # SGDClassifier(loss='log', penalty='l1'),
        SGDClassifier(loss='modified_huber', n_iter=5, random_state=0, shuffle=True),
        LinearSVC(C=1e1),
        LogisticRegressionCV(solver='liblinear'),
        Perceptron(penalty='l2'),
    ]

    global accuracies
    accuracies = {}

    plotting = False
    detailed = False  # to output classifiers parameters info
    if args.plot:
        plotting = True
    if args.details:
        detailed = True
    for c in classifiers:
        classify(c, detailed, plotting)

    info('Embedding vectors: {}\n'.format(model_file))

    # write results to a file
    model_name = model_file.replace('.bin', '').split('/')[-1]
    print(model_name)
    print(accuracies)

    # dump results as json format
    import json
    aggregate_acc = {model_name: accuracies}
    json_result = 'results/sentiment-results.json'
    open(json_result, 'a').write(json.dumps(aggregate_acc, indent=4))
    open(json_result, 'a').write('\n,')

    # dump results to markdown table format
    with open('results/sentiment-results.txt', 'a') as output:
        h = '\n| model |'
        output.write(h)

        for c, result in accuracies.items():
            clf = '{} |'.format(c)
            output.write(clf)

        h2 = '\n|:------|'
        output.write(h2)

        for _, _ in accuracies.items():
            toprule = ':-------------:|'
            output.write(toprule)

        output.write('\n| {} |'.format(model_name))
        for c, result in accuracies.items():
            score, pos, neg = result
            scr = '{}  |'.format(score)
            output.write(scr)

        output.write('\n\n')


if __name__ == '__main__':
    main()
