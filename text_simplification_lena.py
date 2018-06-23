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
from conjugation import convert

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Load Google's pre-trained Word2Vec model.
model = gensim.models.KeyedVectors.load_word2vec_format('./model/GoogleNews-vectors-negative300-SLIM.bin', binary=True)
from nltk import sent_tokenize, word_tokenize, pos_tag

input = 'These theorists were sometimes only loosely affiliated, and some authors point out that the "Frankfurt circle" was neither a philosophical school nor a political group. Nevertheless, they spoke with a common paradigm in mind; they shared the Marxist Hegelian premises and were preoccupied with similar questions.'
# input = 'Her face was a synthesis of perfect symmetry and unusual proportion; he could have gazed at it for hours, trying to locate the source of its fascination.'

# Load ngrams frequenct dictionary
ngrams = pd.read_csv('ngrams.csv')
ngrams = ngrams.drop_duplicates(subset='bigram', keep='first')
ngram_freq_dict = dict(zip(ngrams.bigram, ngrams.freq))


def generate_freq_dict():
    """ Create frequency dictionary based on BROWN corpora. """
    freq_dict = FreqDist()
    for sentence in brown.sents():
        for word in sentence:
            freq_dict[word] += 1
    return freq_dict


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




def check_if_word_fits_the_context(context, token, replacement):
    """ Check if bigram with the replacement exists. """
    # Todo: combine in a single condition
    if (context[0] + ' ' + replacement).lower() in ngram_freq_dict.keys():
        print('replacement', context[0] + ' ' + replacement, ngram_freq_dict[(context[0] + ' ' + replacement).lower()])
        return True
    elif (replacement + ' ' + context[2]).lower() in ngram_freq_dict.keys():
        print('replacement', replacement + ' ' + context[2], ngram_freq_dict[replacement + ' ' + context[2]])
        return True
    else:
        return False


def generate_word2vec_candidates(word, topn=15):
    """ Return top words from word2vec for each word in input. """
    candidates = set()
    # print(word)
    if check_if_replacable(word) and word in model:
        # print(word[0])
        # print(model.most_similar(word[0], topn=topn))
        candidates = [word[0] for word in model.most_similar(word, topn=topn)]

    return candidates


def generate_wordnet_candidates(word):
    """ Generate wordnet candidates for each word in input. """
    candidates = set()
    if check_if_replacable(word):
        for synset in wordnet.synsets(word):
            for lemma in synset.lemmas():
                candidates.add(lemma.name())
            # for lemma in synset.lemmas():
            #     candidates[lemma.name()] = freq_dict.freq(lemma.name())

    return candidates


def check_if_replacable(word):
    """ Check POS and frequency; """
    word_tag = pos_tag([word])
    # print(word, word.istitle())
    if 'NN' in word_tag[0][1] or 'JJ' in word_tag[0][1] or 'VB' in word_tag[0][1]:
        # print('replacable', word_tag)
        return True
    else:
        return False


def simplify(input):
    simplified = ''

    sents = sent_tokenize(input)  # Split by sentences

    final_word = {}
    for sent in sents:
        tokens = word_tokenize(sent)    # Split a sentence by words

        # Rank by frequency
        freqToken = [None]*len(tokens)
        for index, token in enumerate(tokens):
            freqToken[index] = freq_dict.freq(token)
        # print('freqToken = {}'.format(freqToken))

        sortedtokens = [f for (t, f) in sorted(zip(freqToken, tokens))]
        # print(sortedtokens)

        n = int(0.3 * len(tokens))
        # print('n = ' + str(n))

        # 1. Select difficult words
        difficultWords = []
        for i in range(0, n):
            difficultWord = sortedtokens[i]
            difficultWords.append(difficultWord)
            replacement_candidate = {}

            # 2. Generate candidates
            for option in generate_word2vec_candidates(difficultWord):
                replacement_candidate[option] = freq_dict.freq(option)
            for option in generate_wordnet_candidates(difficultWord):
                replacement_candidate[option] = freq_dict.freq(option)

            # 3. Select the candidate with the highest frequency
            if len(replacement_candidate) > 0:
                final_word[difficultWord] = max(replacement_candidate, key=lambda i: replacement_candidate[i])

        output = []
        for token_id in range(len(tokens)):
            token = tokens[token_id]
            if token in difficultWords and token in final_word and token.istitle() is False:
                if token_id != 0 and token_id != len(tokens):
                    fw_in_tense = convert(final_word[token], token)
                    if check_if_word_fits_the_context(tokens[token_id-1:token_id+2], token, fw_in_tense):
                        # output.append(final_word[token])
                        output.append(fw_in_tense)
                    else:
                        output.append(token)
                else:
                    output.append(token)
            else:
                output.append(token)
        print('v1', ' '.join(output))

        for token in tokens:
            if token in difficultWords and token in final_word and token.istitle() is False:  # Replace word if in is difficult and a candidate was found
                fw_in_tense = convert(final_word[token], token)
                # print('tense', final_word[token], token, fw_in_tense)
                # output.append(final_word[token])
                output.append(fw_in_tense)
            else:
                output.append(token)
        print('v2', ' '.join(output))
        simplified += ' '.join(output)

    return simplified

if __name__ == '__main__':
    freq_dict = generate_freq_dict()

    # Generate ppdb candidates:
    # Using lexical thing

    # Choose suitable word:
    # Should be of the same part of speech
    # Should be more frequent that the original word (first do lemmatisation and then check frequency)

    print(simplify(input))
    print(input)

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
# 1. Word2vec suggestions - only for nouns, verbs and adjectives (everything that starts with NN, VB, JJ and istitle == False)
# 2. Check if fit context (ngrams)
