import re
import pandas as pd
import ujson
import _pickle as pickle
from nltk import sent_tokenize, word_tokenize, pos_tag
import pandas as pd
import gensim

from nltk.corpus import brown
from nltk.probability import *
from nltk.corpus import wordnet
from nltk import sent_tokenize, word_tokenize, pos_tag

import logging
from conjugation import convert


def concat(row):
    return str(row['w1']) + ' ' + str(row['w2'])


def generate_freq_dict():
    """ Create frequency dictionary based on BROWN corpora. """
    freq_dict = FreqDist()
    for sentence in brown.sents():
        for word in sentence:
            freq_dict[word] += 1
    return freq_dict


if __name__ == '__main__':

    # # Ngrams
    # with open('clean_ngrams.txt', 'w') as w:
    #     with open('./data/ngrams_words_2.txt') as f:
    #         i = 0
    #         w.write('freq	w1	w2	t1	t2\n')
    #         for line in f:
    #             line = re.sub(' +', '', line)
    #             w.write(line)
    #             i += 1
    #
    # ngrams = pd.read_csv('clean_ngrams.txt', delimiter='\t')
    # print(ngrams.head())
    # ngrams['bigram'] = ngrams.apply(concat, axis=1)
    # ngrams.to_csv('ngrams.csv')
    # d = ngrams.to_dict('dict')
    # for i in d:
    #     print(i)
    #     print(d[i])
    # pickle.dump(d, open('ngrams.pkl', 'wb'))

    # Choose sentences with high number of infrequent words
    freq_dict = generate_freq_dict()
    top_n = 5000
    freq_top_n = sorted(freq_dict.values(), reverse=True)[top_n - 1]

    difficult_sentences = []
    with open('wiki_input_2.txt', 'w') as w:
        with open('./data/simple.aligned') as f:
            number = 0
            for line in f:
                if number < 3000000:
                    tokens = word_tokenize(line.split('\t')[2])
                    score = 0
                    for word in tokens:
                        if word in freq_dict:
                            if freq_dict[word] < freq_top_n:
                                score += 1
                        else:
                            score += 1
                    if score/len(tokens) > 0.4:
                        difficult_sentences.append(line.split('\t')[2])
                        w.write(line.split('\t')[2])
                number += 1
