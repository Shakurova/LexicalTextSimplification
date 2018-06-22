from nltk.corpus import wordnet as wn

# from: https://nlpforhackers.io/convert-words-between-forms/
# and https://stackoverflow.com/questions/35458896/python-map-nltk-stanford-pos-tags-to-wordnet-pos-tags

def penn2morphy(penntag, returnNone=False):
    morphy_tag = {'NN':wn.NOUN, 'JJ':wn.ADJ,
                  'VB':wn.VERB, 'RB':wn.ADV}
    try:
        return morphy_tag[penntag[:2]]
    except:
        return None if returnNone else ''

def convert(word, from_pos, to_pos):
    """ Transform words given from/to POS tags """
    WN_NOUN = 'n'
    WN_VERB = 'v'
    WN_ADJECTIVE = 'a'
    WN_ADJECTIVE_SATELLITE = 's'
    WN_ADVERB = 'r'

    # Convert nltk tags to wordnet tags
    from_pos = penn2morphy(from_pos)
    to_pos = penn2morphy(to_pos)

    synsets = wn.synsets(word, pos=from_pos)

    # Word not found or tag conversion failed
    if not synsets or from_pos == '' or to_pos == '':
        return []

    # Get all lemmas of the word (consider 'a' and 's' equivalent)
    lemmas = [l for s in synsets
              for l in s.lemmas()
              if s.name().split('.')[1] == from_pos
              or from_pos in (WN_ADJECTIVE, WN_ADJECTIVE_SATELLITE)
              and s.name().split('.')[1] in (WN_ADJECTIVE, WN_ADJECTIVE_SATELLITE)]

    # Get related forms
    derivationally_related_forms = [(l, l.derivationally_related_forms()) for l in lemmas]

    # filter only the desired pos (consider 'a' and 's' equivalent)
    related_noun_lemmas = [l for drf in derivationally_related_forms
                           for l in drf[1]
                           if l.synset.name().split('.')[1] == to_pos
                           or to_pos in (WN_ADJECTIVE, WN_ADJECTIVE_SATELLITE)
                           and l.synset.name().split('.')[1] in (WN_ADJECTIVE, WN_ADJECTIVE_SATELLITE)]

    # Extract the words from the lemmas
    words = [l.name() for l in related_noun_lemmas]
    len_words = len(words)

    # Build the result in the form of a list containing tuples (word, probability)
    result = [(w, float(words.count(w)) / len_words) for w in set(words)]
    result.sort(key=lambda w: -w[1])

    # return all the possibilities sorted by probability
    return result