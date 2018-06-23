# import ppdb
# import cPickle as pickle
import ujson
import editdistance

# ppdb_rules = ppdb.load_ppdb('ppdb-2.0-s-all')
#
# # print(type(ppdb_rules))
# # print(ppdb_rules.keys())
# # print(ppdb_rules['reallocations'])
#
# pickle.dump(ppdb_rules, open('ppdb_rules.pkl', 'w'))
#
#
# # # for portuguese
# # ppdb_rules_pt = ppdb_pt.load_ppdb('ppdb-2.0-s-all')
#
#
# # print(ppdb_rules.get_rhs('A'))
#
# # for texts in wikipedia use ppdb to make simplifications
# # for that all rules in ppdb classify as good (simplifying) and bad
# # apply good rules
# # filter wirds derivations (so it the biggest part of the word is the same - bad rule)
#
#
# rules = pickle.load(open('ppdb_rules.pkl'))
#
# ujson.dumps(ppdb_rules, open('ppdb_rules.json', 'w'))


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

    if relation[0] == 'ReverseEntailment':  # don't include
        return True

    if relation[0] == 'Equivalence':

        if (editdistance.eval(exp1[0], exp2[0]) == 1) and (exp1[0][1] == exp2[0][1]):
            return True

        return False


def load_ppdb(path='ppdb-2.0-s-lexical'):
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
            relation = fields[-1].strip().split()

            lhs = tuple(remove_comma_and_article(lhs))
            rhs = tuple(remove_comma_and_article(rhs))

            if len(lhs) == 0 or len(rhs) == 0:
                continue

            # filter out trivial number/gender variations
            if is_trivial(lhs, rhs, relation):
                # print(lhs, rhs, relation)
                continue

            print(lhs, rhs, relation)
            # add rhs to the transformation dictionary
            # ppdb_rules(lhs, rhs)
            ppdb_rules[rhs] = lhs

    return ppdb_rules


if __name__ == '__main__':
    rules = load_ppdb(path='ppdb-2.0-s-lexical')
    print(rules)
    print(type(rules))
    ujson.dump(rules, open('ppdb_lexical_rules.json', 'w'))


# To-do:
# Approach 1:
#   Replacing complex words with their most frequent synonym.

# Select all candidates;
# Explicit sense labelling;
# Implicit sense labelling;
# Part-of-speech tag ltering; and
# Semantic similarity ltering.

# find a corpora with synonyms - filter on how close they are (look at presentation slide I sent on telegram)
