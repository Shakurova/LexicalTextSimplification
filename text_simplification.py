""" Simple text simplification approach based on frequency.
Choose 30% top low frequent words in the sentence,
replace them with the most frequent candidate from wordnet """

import pickle
import operator
from operator import itemgetter
import re

from nltk.corpus import brown
from nltk.probability import *
from nltk.corpus import wordnet
from nltk import sent_tokenize, word_tokenize, pos_tag
from functions import convert

input = 'Her face was a synthesis of perfect symmetry and unusual proportion; he could have gazed at it for hours, trying to locate the source of its fascination.'


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
                fw_in_tense = convert(final_word[token],pos_tag([final_word[token]])[0][1],pos_tag([token])[0][1])
                if fw_in_tense == []:
                    output.append(final_word[token])
                else:
                    output.append(fw_in_tense[0][0]) # print(final_word[token])
            else:
                output.append(token) #print(token)
        print(output)


if __name__ == '__main__':
    freq_dict = generate_freq_dict()
    #print(freq_dict)

    frequency_approach(freq_dict, input)


# Todo:
# choose complex word (long and not frequent)
# choose candidates - from dictionary of synonyms abd top word2vec words - check for gender, tense
# choose suitable - those who occur in ngram dictionary

# Am I using paraphrasing?

# a lot of functions from here
# https://github.com/SIMPATICOProject/SimpaticoTAEServer/blob/master/lexical_simplification_server/lib.py

# 1. ComplexWordIdentifier
# 2. Generator (generate suitable candidates)
# 3. Ranker
