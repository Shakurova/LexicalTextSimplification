""" Simple text simplification approach based on frequency.
Choose 30% top low frequent words in the sentence,
replace them with the most frequent candidate from wordnet """

import operator
from operator import itemgetter
import re
import pandas as pd

import ujson

from nltk.corpus import brown
from nltk.probability import *
from nltk.corpus import wordnet
from nltk import sent_tokenize, word_tokenize, pos_tag
from nltk import sent_tokenize, word_tokenize
import gensim
import _pickle as pickle


import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Load Google's pre-trained Word2Vec model.
model = gensim.models.KeyedVectors.load_word2vec_format('./model/GoogleNews-vectors-negative300-SLIM.bin', binary=True)
from nltk import sent_tokenize, word_tokenize, pos_tag
from functions import convert

input = 'Her face was a synthesis of perfect symmetry and unusual proportion; he could have gazed at it for hours, trying to locate the source of its fascination.'

# Load ngrams frequenct dictionary
ngrams = pd.read_csv('ngrams.csv')
area_dict = dict(zip(ngrams.bigram, ngrams.freq))
# print(area_dict)


def generate_freq_dict():
    """ Create frequency dictionary based on BROWN corpora. """
    freq_dict = FreqDist()
    for sentence in brown.sents():
        for word in sentence:
            freq_dict[word] += 1
    return freq_dict


def frequency_approach(freq_dict, input):
    sents = sent_tokenize(input)  # Split by sentences

    final_word = {}
    for sent in sents:
        tokens = word_tokenize(sent)    # Split a sentence by words

        # Rank by frequency
        freqToken = [None]*len(tokens)
        for index, token in enumerate(tokens):
            freqToken[index] = freq_dict.freq(token)
        print('freqToken = {}'.format(freqToken))

        sortedtokens = [f for (t, f) in sorted(zip(freqToken, tokens))]
        print(sortedtokens)

        n = int(0.3 * len(tokens))
        #print('n = ' + str(n))

        # 1. Select difficult words
        difficultWords = []
        for i in range(0, n):
            difficultWord = sortedtokens[i]
            difficultWords.append(difficultWord)
            replacement_candidate = {}

            # 2. Generate candidates
            for synset in wordnet.synsets(difficultWord):
                for lemma in synset.lemmas():
                    replacement_candidate[lemma.name()] = freq_dict.freq(lemma.name())

            # 3. Select the candidate with the highest frequency
            if len(replacement_candidate) > 0:
                final_word[difficultWord] = max(replacement_candidate, key=lambda i: replacement_candidate[i])

        output = []
        for token in tokens:
            if token in difficultWords and token in final_word:  # replace word if in is difficult and a candidate was found
                output.append(final_word[token])
            else:
                output.append(token)
        print(output)

        # # Jelmer tense thingie
        # output = []
        # for token in tokens:
        #     if token in difficultWords and token in final_word:  # replace word if in is difficult and a candidate was found
        #         fw_in_tense = convert(final_word[token], pos_tag([final_word[token]])[0][1], pos_tag([token])[0][1])
        #         if fw_in_tense == []:
        #             output.append(final_word)
        #         else:
        #             output.append() # print(final_word[token])
        #     else:
        #         output.append(token) #print(token)
        #
        # print(output)


def check_length(word):
    """ Return word length. """
    return len(word)


def check_frequency(word):
    """ Return word frequency. """
    return freq_dict[word]


def return_synonyms(word):
    """ Return synonyms from wordnet. """
    replacement_candidate = []
    for synset in wordnet.synsets(word):
        for lemma in synset.lemmas():
            replacement_candidate.append(lemma.name())
    return replacement_candidate


def check_if_word_fits_the_context(word1, word2, left, right):
    """ Check if bigrm with the replacement exists. """
    ############# TEST ##################
    if left + ' ' + word2 in ngrams and word2 + ' ' + right in ngrams:
        return True
    else:
        return False


def generate_word2vec_candidates(input, topn=10):
    """ Return top words from word2vec for each word in input. """
    candidates = {}
    for word in pos_tag(word_tokenize(input)):
        if word[0] in model and ('NN' in word[1] or 'JJ' in word[1] or 'VB' in word[1]):
            # print(word[0])
            # print(model.most_similar(word[0], topn=topn))
            candidates[word[0]] = [word[0] for word in model.most_similar(word[0], topn=topn)]

    return candidates


def generate_wordnet_candidates(input):
    """ Generate wordnet candidates for each word in input. """
    candidates = {}
    for word in word_tokenize(input):
        if check_if_replacable(word):
            candidates[word] = set()
            for synset in wordnet.synsets(word):
                for lemma in synset.lemmas():
                    candidates[word].add(lemma.name())
                # for lemma in synset.lemmas():
                #     candidates[lemma.name()] = freq_dict.freq(lemma.name())

    return candidates


def check_if_replacable(word):
    """ Check POS and frequency; """
    word_tag = pos_tag([word])
    if 'NN' in word_tag[0][1] or 'JJ' in word_tag[0][1] or 'VB' in word_tag[0][1] and freq_dict.freq(word) > 0.5:
        return True
    else:
        return False

if __name__ == '__main__':
    freq_dict = generate_freq_dict()
    # print(freq_dict)

    frequency_approach(freq_dict, input)

    # Generate word2vec candidates:
    print(generate_word2vec_candidates(input))
    print(generate_wordnet_candidates(input))

    # Choose suitable word:
    # Should be of the same part of speech
    # Should be more frequent that the original word (first do lemmatisation and then check frequency)

# Todo:
# choose complex word (long and not frequent)
# choose candidates - from dictionary of synonyms abd top word2vec words - check for gender, tense
# choose suitable - those who occur in ngram dictionary

# Am I using ppdb?

# a lot of functions from here
# https://github.com/SIMPATICOProject/SimpaticoTAEServer/blob/master/lexical_simplification_server/lib.py

# 1. ComplexWordIdentifier
# 2. Generator (generate suitable candidates)
# 3. Ranker

# Word2vec, synonyms, ppdb

# Use ngrams to check the context

# Today:
# 1. Word2vec suggestions - only for nouns, verbs and adjectives (everything that starts with NN, VB, JJ)
# 2. Check if fit context (ngrams)
