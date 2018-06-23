from nltk.corpus import wordnet as wn
from pattern.en import referenced
from pattern.en import pluralize, singularize
from pattern.en import conjugate, lemma, lexeme
from pattern.en import tag
from pattern.en import parse

# https://www.clips.uantwerpen.be/pages/pattern-en#conjugation


def convert(word_from, word_to):
    """ Analyses POS tags and converts words to a desired form. """
    # Todo: NOUN to Adjective conversion

    # print(tag(word_to))
    # print(tag(word_from))
    if tag(word_to)[0][1] == 'VBD':
        converted = conjugate(word_from, 'past')
    elif tag(word_to)[0][1] == 'VBN':
        converted = conjugate(word_from, 'past')
    elif tag(word_to)[0][1] == 'VBZ':
        converted = conjugate(word_from, 'present', '3sg')
    elif tag(word_to)[0][1] == 'VBP':
        converted = conjugate(word_from, 'present', '1sg')
    elif tag(word_to)[0][1] == 'VB':
        converted = conjugate(word_from, 'infinitive')
    elif tag(word_to)[0][1] == 'NN' and tag(word_from)[0][1] == 'NNS':
        converted = singularize(word_from)
    elif tag(word_to)[0][1] == 'NNS' and tag(word_from)[0][1] == 'NN':
        converted = pluralize(word_from)
    else:
        converted = word_from

    return converted

if __name__ == "__main__":
    print(convert('sing', 'called'))
    print(convert('ball', 'cats'))
    print(convert('gazed', 'stare'))
    print(convert('stared', 'gaze'))