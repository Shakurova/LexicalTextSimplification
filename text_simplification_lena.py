""" Simple text simplification approach based on frequency.
Choose 30% top low frequent words in the sentence,
replace them with the most frequent candidate from wordnet """

import pandas as pd
import gensim

from nltk.corpus import brown
from nltk.probability import *
from nltk.corpus import wordnet
from nltk import sent_tokenize, word_tokenize, pos_tag

import logging
from conjugation import convert

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Load Google's pre-trained Word2Vec model.
model = gensim.models.KeyedVectors.load_word2vec_format('./model/GoogleNews-vectors-negative300-SLIM.bin', binary=True)


# Load ngrams frequenct dictionary
ngrams = pd.read_csv('ngrams.csv')
ngrams = ngrams.drop_duplicates(subset='bigram', keep='first')
ngram_freq_dict = dict(zip(ngrams.bigram, ngrams.freq))


def generate_freq_dict(self):
    """ Create frequency dictionary based on BROWN corpora. """
    freq_dict = FreqDist()
    for sentence in brown.sents():
        for word in sentence:
            freq_dict[word] += 1
    return freq_dict


def check_if_word_fits_the_context(context, token, replacement):
    """ Check if bigrm with the replacement exists. """
    # Todo: combine in a single condition
    print('context', context)
    if (context[0] + ' ' + replacement).lower() in ngram_freq_dict.keys():
        print('replacement', context[0] + ' ' + replacement, ngram_freq_dict[(context[0] + ' ' + replacement).lower()])
        return True
    if (replacement + ' ' + context[2]).lower() in ngram_freq_dict.keys():
        print('replacement', replacement + ' ' + context[2], ngram_freq_dict[(replacement + ' ' + context[2]).lower()])
        return True
    else:
        return False


def return_bigram_score(context, token, replacement):
    score = 0
    if (context[0] + ' ' + replacement).lower() in ngram_freq_dict.keys():
        print('score', context[0] + ' ' + replacement, ngram_freq_dict[(context[0] + ' ' + replacement).lower()])
        score += ngram_freq_dict[(context[0] + ' ' + replacement).lower()]
    if (replacement + ' ' + context[2]).lower() in ngram_freq_dict.keys():
        print('score', replacement + ' ' + context[2], ngram_freq_dict[(replacement + ' ' + context[2]).lower()])
        score += ngram_freq_dict[(replacement + ' ' + context[2]).lower()]
    return score / 2


def generate_word2vec_candidates(word, topn=15):
    """ Return top words from word2vec for each word in input. """
    candidates = set()
    # print(word)
    if check_if_replacable(word) and word in model:
        # print(word[0])
        # print(model.most_similar(word[0], topn=topn))
        # candidates = [option[0].lower() for option in model.most_similar(word, topn=topn) if word != option[0]]
        candidates = [convert(option[0].lower(), word) for option in model.most_similar(word, topn=topn)
                      if convert(option[0].lower(), word) != word and convert(option[0].lower(), word) != None]

    return candidates


def generate_wordnet_candidates(word):
    """ Generate wordnet candidates for each word in input. """
    candidates = set()
    if check_if_replacable(word):
        for synset in wordnet.synsets(word):
            for lemma in synset.lemmas():
                if convert(lemma.name().lower(), word) != word and convert(lemma.name().lower(), word) != None:
                    candidates.add(convert(lemma.name().lower(), word))

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
    simplified0 = ''
    simplified1 = ''
    simplified2 = ''

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
        all_options = {}
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

            all_options[difficultWord] = replacement_candidate  # keep all the versions
            # 3. Select the candidate with the highest frequency
            if len(replacement_candidate) > 0:
                final_word[difficultWord] = max(replacement_candidate, key=lambda i: replacement_candidate[i])
                print('!!!replacement_candidate', replacement_candidate)

        # Keep only suitable candidates
        best_candidates = {}
        print('all options', all_options)
        for token_id in range(len(tokens)):
            token = tokens[token_id]
            best_candidates[token] = {}
            if token in all_options:
                print(token, ':', all_options[token])
                # options[token] = list of options
                # iterate over list of options
                for opt in all_options[token]:
                    fw_in_tense = convert(opt, token)
                    print('option: ', opt)
                    if token_id != 0 and token_id != len(tokens):
                        if check_if_word_fits_the_context(tokens[token_id - 1:token_id + 2], token, fw_in_tense):
                            print('GOOD', tokens[token_id - 1:token_id + 2], token, opt, fw_in_tense)
                            print(return_bigram_score(tokens[token_id - 1:token_id + 2], token, fw_in_tense))
                            # best_candidates[token][fw_in_tense] = return_bigram_score(tokens[token_id - 1:token_id + 2], token, fw_in_tense)
                            best_candidates[token][fw_in_tense] = return_bigram_score(tokens[token_id - 1:token_id + 2], token, fw_in_tense)
                        else:
                            print('BAD', tokens[token_id - 1:token_id + 2], token, opt, fw_in_tense)
                            print(return_bigram_score(tokens[token_id - 1:token_id + 2], token, fw_in_tense))
                # check_if_word_fits_the_context returns average frequency - then take the bigram with the highest average frequency

        print('BEST CANDIDATES', best_candidates)

        # Generate replacements0
        output = []
        for word in tokens:
            if word in best_candidates:
                # for word in best_candidates:
                if word.istitle() is False and best_candidates[word]!= {}:
                    print('best_candidates[word]', word, best_candidates[word])
                    print(max(best_candidates[word], key=lambda i: best_candidates[word][i]))
                    output.append(max(best_candidates[word], key=lambda i: best_candidates[word][i]))
                else:
                    output.append(word)
            else:
                output.append(word)
        print('v0', ' '.join(output))
        simplified0 += ' '.join(output)

        # Generate replacements1
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
        simplified1 += ' '.join(output)

        # Generate replacements2
        output = []
        for token in tokens:
            if token in difficultWords and token in final_word and token.istitle() is False:  # Replace word if in is difficult and a candidate was found
                fw_in_tense = convert(final_word[token], token)
                # print('tense', final_word[token], token, fw_in_tense)
                # output.append(final_word[token])
                output.append(fw_in_tense)
            else:
                output.append(token)
        print('v2', ' '.join(output))
        simplified2 += ' '.join(output)

    return simplified0, simplified1, simplified2

if __name__ == '__main__':
    freq_dict = generate_freq_dict()

    # Generate ppdb candidates:
    # Using lexical thing

    # Choose suitable word:
    # Should be of the same part of speech
    # Should be more frequent that the original word (first do lemmatisation and then check frequency)

    with open('testset.txt') as f:
        with open('output.txt', 'w') as w:
            for input in f:
                simplified0, simplified1, simplified2 = simplify(input)
                print('original', input)
                w.write(simplified0 + '\t' + simplified1 + '\t' + simplified2 + '\n')

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
#
