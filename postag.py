import nltk
from nltk.corpus import brown
import pickle
import sys

# 1

brown_tagged_sents = brown.tagged_sents(categories='news')
train = nltk.UnigramTagger(brown_tagged_sents)


# 2

sentence = "Gone with the wind."
# print(train.tag(sentence.split()))



# 3

test_news = train.evaluate(brown.tagged_sents(categories='news'))

# 4

test_reviews = train.evaluate(brown.tagged_sents(categories='reviews'))


def get_data(arg):
    return brown.tagged_sents(categories=arg)

def tagger_training(data):
    return nltk.UnigramTagger(data)


