# import ppdb
# import cPickle as pickle
import os
import pickle
# import ujson
import editdistance
import nltk

def remove_comma_and_article(expression):
    """
    Filter an expression by removing any leading articles and/or commas.

    :param expression: a list/tuple of strings
    :return: a list of strings
    """

    articles = {'a', 'the'}
    if len(expression) == 1:
        return expression

    while expression[0] in articles or expression[0] == ',':
        expression = expression[1:]
        if len(expression) == 0:
            return expression

    if expression[-1] == ',':
        expression = expression[:-1]

    # here check that left is more frequent than right

    return expression


def is_trivial(exp1, exp2, relation):
    """
    Return True if:
        - w1 and w2 differ only in gender and/or number
        - w1 and w2 differ only in a heading preposition

    :param exp1: tuple/list of strings, expression1
    :param exp2: tuple/list of strings, expression2
    :return: boolean
    """

    endings = ['ing', 'ed', 's', 'ic']
    ps = nltk.PorterStemmer()

    # Contains many false positives
    if relation == 'ReverseEntailment':  # don't include
        return True

    # if relation == 'Equivalence':
        # Synonyms are unlikely to be 1 symbol apart, but are likely to be US/UK spelling differences we do not want
    if (editdistance.eval(exp1[0], exp2[0]) == 1):
        return True


    if len(exp1) == 1 and len(exp2) == 1 and ps.stem(exp1[0]) == ps.stem(exp2[0]):
        return True

    return False


def load_ppdb(path='.data/ppdb-2.0-s-lexical', load_pickle = True):
    PICKLE_PATH = './data/ppdb_lexical_rules.pkl'
    if load_pickle and os.path.isfile(PICKLE_PATH):
        return pickle.load(open(PICKLE_PATH, 'rb'))
    else:
        ppdb_rules = {}
        with open(path, 'r') as f:
            for line in f:
                # line = line.decode('utf-8')
                # discard lines with unrecoverable encoding errors
                if '\\ x' in line or 'xc3' in line:
                    continue
                fields = line.split('|||')
                lhs = fields[1].strip().split()
                rhs = fields[2].strip().split()
                relation = fields[-1].strip()

                lhs = tuple(remove_comma_and_article(lhs))
                rhs = tuple(remove_comma_and_article(rhs))

                if len(lhs) == 0 or len(rhs) == 0:
                    continue

                # filter out trivial number/gender variations
                if is_trivial(lhs, rhs, relation):
                    # print(lhs, rhs, relation)
                    continue

                # print(lhs, rhs, relation)
                # add rhs to the transformation dictionary
                # ppdb_rules(lhs, rhs)
                ppdb_rules[lhs[0]] = rhs[0]
        print("Nr of rules: ", len(ppdb_rules))
        pickle.dump(ppdb_rules, open(PICKLE_PATH, 'wb'))
        return ppdb_rules


if __name__ == '__main__':
    rules = load_ppdb(path='./data/ppdb-2.0-s-all', load_pickle=False)
    input = 'These theorists were sometimes only loosely affiliated, and some authors point out that the "Frankfurt circle" was neither a philosophical school nor a political group.'

    for word in input.split():
        if word in rules:
            print(word + " -> "  + rules[word])


# To-do:
# Approach 1:
#   Replacing complex words with their most frequent synonym.

# Select all candidates;
# Explicit sense labelling;
# Implicit sense labelling;
# Part-of-speech tag ltering; and
# Semantic similarity ltering.

# find a corpora with synonyms - filter on how close they are (look at presentation slide I sent on telegram)
