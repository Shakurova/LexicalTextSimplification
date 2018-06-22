import re
import pandas as pd
import ujson
import _pickle as pickle

with open('clean_ngrams.txt', 'w') as w:
    with open('./data/ngrams_words_2.txt') as f:
        i = 0
        w.write('freq	w1	w2	t1	t2\n')
        for line in f:
            line = re.sub(' +', '', line)
            w.write(line)
            i += 1


ngrams = pd.read_csv('clean_ngrams.txt', delimiter='\t')
print(ngrams.head())


def concat(row):
    return str(row['w1']) + ' ' + str(row['w2'])

ngrams['bigram'] = ngrams.apply(concat, axis=1)

ngrams.to_csv('ngrams.csv')

d = ngrams.to_dict('dict')

for i in d:
    print(i)
    print(d[i])

pickle.dump(d, open('ngrams.pkl', 'wb'))