# LexicalTextSimplification
Lexical Text Simplification for Cognitive Computational Modeling of Language and Web Interaction course.

Similar to most lexical simplification systems, the lexical simplification module of our system first has to identify complex words, then generate their substitutes, before then filtering and ranking these to determine the best replacement.

For complex word identification we take two approaches - we either choose 30% of least frequent words in the sentence (frequency is calculated on BROWN corpora) or any words that are not in the top 3000 most frequent words in our corpora.

For our substitute generation, we use Wordnet synonyms and top 15 word2vec candidates.

For filtering we check whether the part-of-speech tag of the candidate is the same as that of the original word, and whether the new word fits the context (using bigram corpora), we do not substitute words that start with capital letter.

For ranking we either use frequency or bigram score which is the averaged frequency of left and right-context bigram. The bigrams were taken from the Corpus of Contemporary American English \cite{cca}.

In the end, we choose the most suitable word and convert it to the same form as the original word (tense, plural/singular) using pattern library (reference).

##  HOW RO RUN

Code is written in Python 3.5. You need to install [Pattern library](https://github.com/clips/pattern/tree/python3) Python3 branch

```
pip install -r requirements.txt
```

## Datasets used

BROWN corpora
Ngrams corpora


## Results

Lexical text simplification - verion 1, verion 2, verion 3
```
best v1:impression -> idea
best v2:impression -> sense
best v3:impression -> sense

```

Lexical text simplification - verion 1, verion 2, verion 3
```
original: Nevertheless, they spoke with a common paradigm in mind; they shared the Marxist Hegelian premises and were preoccupied with similar questions.
v0 Nevertheless , they spoke with a common image in mind ; they embraced the Marxist Hegelian assumptions and were obsessed with similar questions .
v1 Nevertheless , they spoke with a common image in mind ; they expressed the Marxist Hegelian assumptions and were lost with similar questions .
v2 Nevertheless , they spoke with a common image in mind ; they expressed the Marxist Hegelian assumptions and were lost with similar questions .
```

We also wrote a converter using [Pattern library](https://github.com/clips/pattern/tree/python3) that converts the replacement word to the same morphological word form as the original word.
```
clause -> articles = clauses
blog -> articles = blogs
expend -> used = expended
apply -> used = applied
```


