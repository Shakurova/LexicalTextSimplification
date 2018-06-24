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

# Load Google's pre-trained Word2Vec model
model = gensim.models.KeyedVectors.load_word2vec_format('./model/GoogleNews-vectors-negative300-SLIM.bin', binary=True)

# Load ngrams frequency dictionary
ngrams = pd.read_csv('ngrams.csv')
ngrams = ngrams.drop_duplicates(subset='bigram', keep='first')

steps = open('steps.txt', 'w')


def generate_freq_dict():
    """ Create frequency dictionary based on BROWN corpora. """
    freq_dict = FreqDist()
    for sentence in brown.sents():
        for word in sentence:
            freq_dict[word] += 1
    return freq_dict


class Simplifier:
    def __init__(self):
        self.ngram_freq_dict = dict(zip(ngrams.bigram, ngrams.freq))
        self.freq_dict = generate_freq_dict()

    def check_if_word_fits_the_context(self, context, token, replacement):
        """ Check if bigram with the replacement exists. """
        # Todo: combine in a single condition
        if (context[0] + ' ' + replacement).lower() in self.ngram_freq_dict.keys():
            return True
        if (replacement + ' ' + context[2]).lower() in self.ngram_freq_dict.keys():
            return True
        else:
            return False

    def return_bigram_score(self, context, replacement):
        """ Return ad averaged frequency of left- and right-context bigram. """
        # Todo: incorporate word2vec value
        score = 0
        if (context[0] + ' ' + replacement).lower() in self.ngram_freq_dict.keys():
            score += self.ngram_freq_dict[(context[0] + ' ' + replacement).lower()]
        if (replacement + ' ' + context[2]).lower() in self.ngram_freq_dict.keys():
            score += self.ngram_freq_dict[(replacement + ' ' + context[2]).lower()]
        return score / 2

    def generate_word2vec_candidates(self, word, topn=15):
        """ Return top words from word2vec for each word in input. """
        candidates = set()
        if self.check_if_replacable(word) and word in model:
            candidates = [convert(option[0].lower(), word) for option in model.most_similar(word, topn=topn)
                          if convert(option[0].lower(), word) != word and convert(option[0].lower(), word) != None]

        return candidates

    def generate_wordnet_candidates(self, word):
        """ Generate wordnet candidates for each word in input. """
        candidates = set()
        if self.check_if_replacable(word):
            for synset in wordnet.synsets(word):
                for lemma in synset.lemmas():
                    if convert(lemma.name().lower(), word) != word and convert(lemma.name().lower(), word) != None:
                        candidates.add(convert(lemma.name().lower(), word))

        return candidates

    def check_if_replacable(self, word):
        """ Check POS, we only want to replace nouns, adjectives and verbs. """
        word_tag = pos_tag([word])
        if 'NN' in word_tag[0][1] or 'JJ' in word_tag[0][1] or 'VB' in word_tag[0][1]:
            return True
        else:
            return False

    def pick_tokens_by_proportion(self, tokens, threshold=0.3):
        """ threshold - Proportion of words in a sentence to replace (rounded down). """
        # Rank by frequency
        freqToken = [None] * len(tokens)
        for index, token in enumerate(tokens):
            freqToken[index] = self.freq_dict.freq(token)
        sortedtokens = [f for (t, f) in sorted(zip(freqToken, tokens))]
        return sortedtokens[:int(0.3 * len(tokens))] # take top 30% of unfrequent words

    def simplify(self, input):
        simplified0 = ''
        simplified1 = ''
        simplified2 = ''

        sents = sent_tokenize(input)  # Split by sentences

        # Top N most frequent words we never replace
        # top_n = 3000
        # freq_top_n = sorted(self.freq_dict.values(), reverse=True)[top_n - 1]

        for sent in sents:
            steps.write(sent + '\n')
            tokens = word_tokenize(sent)  # Split a sentence by words

            # Replace only words that are not in the top_n most frequent in the dictionary
            # difficultWords = [t for t in tokens if self.freq_dict[t] < freq_top_n]

            # 1. Find difficult words
            difficultWords = self.pick_tokens_by_proportion(tokens, 0.3)

            steps.write('difficultWords:' + str(difficultWords) + '\n')

            all_options = {}
            for difficultWord in difficultWords:
                replacement_candidate = {}

                # 2. Generate candidates
                for option in self.generate_word2vec_candidates(difficultWord):
                    replacement_candidate[option] = self.freq_dict.freq(option)
                for option in self.generate_wordnet_candidates(difficultWord):
                    replacement_candidate[option] = self.freq_dict.freq(option)

                # 2.1. Replacement options with frequency
                all_options[difficultWord] = replacement_candidate
            steps.write('all_options:' + str(all_options) + '\n')

            # 2.2. Replacement options with bigram score
            best_candidates = {}
            for token_id in range(len(tokens)):
                token = tokens[token_id]
                best_candidates[token] = {}
                if token in all_options:
                    for opt in all_options[token]:
                        if token_id != 0 and token_id != len(tokens):  # if not the first or the last word in the sentence
                            if self.check_if_word_fits_the_context(tokens[token_id - 1:token_id + 2], token, opt):
                                # Return all candidates with its bigram scores
                                best_candidates[token][opt] = self.return_bigram_score(tokens[token_id - 1:token_id + 2], token, opt)
            steps.write('best_candidates:' + str(best_candidates) + '\n')

            # 3. Generate replacements0 - take the word with the highest bigram score
            output = []
            for token in tokens:
                if token in best_candidates:
                    if token.istitle() is False and best_candidates[token] != {}:
                        # Choose the one with the highest bigram score
                        best = max(best_candidates[token], key=lambda i: best_candidates[token][i])
                        steps.write('best v1:' + str(token) + ' -> ' + str(best) + '\n')
                        output.append(best)
                    else:
                        output.append(token)
                else:
                    output.append(token)
            print('v0', ' '.join(output))
            simplified0 += ' '.join(output)

            # 3. Generate replacements1 - take the word with the highest frequency + check the context
            output = []
            for token_id in range(len(tokens)):
                token = tokens[token_id]
                if token in all_options and len(all_options[token]) > 0 and token in difficultWords and token.istitle() is False:
                    if token_id != 0 and token_id != len(tokens):
                        # Choose most frequent
                        best = max(all_options[token], key=lambda i: all_options[token][i])
                        steps.write('best v2:' + str(token) + ' -> ' + str(best) + '\n')
                        if self.check_if_word_fits_the_context(tokens[token_id - 1:token_id + 2], token, best):
                            output.append(best)
                        else:
                            output.append(token)
                    else:
                        output.append(token)
                else:
                    output.append(token)
            print('v1', ' '.join(output))
            simplified1 += ' '.join(output)

            # 3. Generate replacements2  - take the word with the highest frequency
            output = []
            for token in tokens:
                # Replace word if in is difficult and a candidate was found
                if token in all_options and len(all_options[token]) > 0 and token in difficultWords and token.istitle() is False:
                    best = max(all_options[token], key=lambda i: all_options[token][i])
                    steps.write('best v3:' + str(token) + ' -> ' + str(best) + '\n')
                    output.append(best)
                else:
                    output.append(token)
            print('v2', ' '.join(output))
            simplified2 += ' '.join(output)

        return simplified0, simplified1, simplified2


if __name__ == '__main__':
    simplifier = Simplifier()

    with open('input.txt') as f:
        with open('output2.csv', 'w') as w:
            for input in f:
                simplified0, simplified1, simplified2 = simplifier.simplify(input)
                print('Original', input)
                w.write(simplified0 + '\t' + simplified1 + '\t' + simplified2 + '\n')

# 1. Choose difficult words (long and not frequent)
# 2. Generate candidates - from dictionary of synonyms abd top word2vec words - check for gender, tense
# 3. Choose the best candidate - check the context and frequency
