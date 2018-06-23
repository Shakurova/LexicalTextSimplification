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

    def return_bigram_score(self, context, token, replacement):
        """ Return ad averaged frequency of left- and right-context bigram. """
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

    def simplify(self, input):
        simplified0 = ''
        simplified1 = ''
        simplified2 = ''

        sents = sent_tokenize(input)  # Split by sentences

        final_word = {}
        for sent in sents:
            tokens = word_tokenize(sent)  # Split a sentence by words

            # Rank by frequency
            freqToken = [None] * len(tokens)
            for index, token in enumerate(tokens):
                freqToken[index] = self.freq_dict.freq(token)

            sortedtokens = [f for (t, f) in sorted(zip(freqToken, tokens))]

            n = int(0.3 * len(tokens))

            # 1. Select difficult words
            all_options = {}
            difficultWords = []
            for i in range(0, n):
                difficultWord = sortedtokens[i]
                difficultWords.append(difficultWord)
                replacement_candidate = {}

                # 2. Generate candidates
                for option in self.generate_word2vec_candidates(difficultWord):
                    replacement_candidate[option] = self.freq_dict.freq(option)
                for option in self.generate_wordnet_candidates(difficultWord):
                    replacement_candidate[option] = self.freq_dict.freq(option)

                all_options[difficultWord] = replacement_candidate  # keep all the versions
                # 3. Select the candidate with the highest frequency
                if len(replacement_candidate) > 0:
                    final_word[difficultWord] = max(replacement_candidate, key=lambda i: replacement_candidate[i])

            # Keep only suitable candidates
            best_candidates = {}
            for token_id in range(len(tokens)):
                token = tokens[token_id]
                best_candidates[token] = {}
                if token in all_options:
                    for opt in all_options[token]:
                        fw_in_tense = convert(opt, token)
                        if token_id != 0 and token_id != len(tokens):
                            if self.check_if_word_fits_the_context(tokens[token_id - 1:token_id + 2], token, fw_in_tense):
                                best_candidates[token][fw_in_tense] = self.return_bigram_score(tokens[token_id - 1:token_id + 2], token, fw_in_tense)

            # Generate replacements0
            output = []
            for word in tokens:
                if word in best_candidates:
                    if word.istitle() is False and best_candidates[word] != {}:
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
                        if self.check_if_word_fits_the_context(tokens[token_id - 1:token_id + 2], token, fw_in_tense):
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
                    output.append(fw_in_tense)
                else:
                    output.append(token)
            print('v2', ' '.join(output))
            simplified2 += ' '.join(output)

        return simplified0, simplified1, simplified2


if __name__ == '__main__':
    simplifier = Simplifier()

    with open('testset.txt') as f:
        with open('output.txt', 'w') as w:
            for input in f:
                simplified0, simplified1, simplified2 = simplifier.simplify(input)
                print('Original', input)
                w.write(simplified0 + '\t' + simplified1 + '\t' + simplified2 + '\n')
