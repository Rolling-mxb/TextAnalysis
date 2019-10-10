"""main.py

Code scaffolding

"""

import os
import nltk
from nltk.corpus import brown
from nltk.corpus import wordnet as wn
from nltk.corpus import PlaintextCorpusReader
from nltk.probability import FreqDist
from nltk.text import Text
from nltk.corpus import stopwords
import string
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.tokenize import RegexpTokenizer
import itertools
import numpy as np

def read_text(path):
    if os.path.isdir(path):
        files = []
        for i in os.listdir(path):
            file = open(os.path.join(path,i), 'r')
            files.append(file.read())
        files = '\n'.join(files)
        tokens = nltk.word_tokenize(files)
        text = nltk.Text(tokens)
        return text
    if os.path.isfile(path):
        files = open(path)
        raw = files.read()
        tokens = nltk.word_tokenize(raw)
        text = nltk.Text(tokens)
        return text


emma = read_text('data/emma.txt')

def token_count(text):
    return len(text)



def type_count(text):
    aa = [x.lower() for x in text.tokens]
    return len(set(aa))


def sentence_count(text):
    aa = TreebankWordDetokenizer().detokenize(text.tokens)
    return len(nltk.sent_tokenize(aa))


def most_frequent_content_words(text):
    stop_words = set(stopwords.words('english'))
    filtered_stopwords = [w for w in text.tokens if not w.lower() in stop_words]
    filtered_all = [word for word in filtered_stopwords if word[0].isalpha()]
    fdist = FreqDist(filtered_all)
    tops = fdist.most_common()[:24]
    return tops
# must have a wrong way of filtering punctuation(can't keep words like 'Mr.')

def most_frequent_bigrams(text):
    stop_words = set(stopwords.words('english'))
    filtered_stopwords = [w for w in text.tokens if not w.lower() in stop_words]
    filtered_all = [word for word in filtered_stopwords if word[0].isalpha()]
    bgs = nltk.bigrams(filtered_all)
    fdist = nltk.FreqDist(bgs)    
    biagrams = []
    for k,v in fdist.items():
        biagrams.append((k,v))
    freq = sorted(biagrams, key = lambda x: x[1], reverse = True)[:24]
    return freq



class Vocabulary():

    def __init__(self, text):
        self.text = text

    def frequency(self, word):
        fdist = FreqDist(text.tokens)
        d = dict(fdist)
        return d[word]

    def pos(self, word):
        d = dict(nltk.pos_tag(text.tokens))
        return d[word]

    def gloss(self, word):
        wn_lemmas = set(wn.all_lemma_names())
        if word in wn_lemmas:
            syn = wn.synsets(word)[0]
            return syn.definition()
        else:
            return None

    def kwic(self, word):
        return text.concordance(word)


categories = ('adventure', 'fiction', 'government', 'humor', 'news')


def compare_to_brown(text):

    # take all the words in the text and create frequency dictionary
    a = FreqDist(text.tokens)
    text_dic = dict(a.most_common())
    # take all the words from each category and do the same above
    brown_text = []
    for i in categories:
        brown_text.append(brown.words(categories=i))

    adventure_dic = dict(FreqDist(brown_text[0]).most_common())
    fiction_dic = dict(FreqDist(brown_text[1]).most_common())
    government_dic = dict(FreqDist(brown_text[2]).most_common())
    humor_dic = dict(FreqDist(brown_text[3]).most_common())
    news_dic = dict(FreqDist(brown_text[4]).most_common())

    # crate vocabulary (union approach)
    vocal = set(list(itertools.chain.from_iterable(brown_text)))

    # calculate similarity

    vtext = np.zeros(len(vocal), dtype = int)
    vadventure = np.zeros(len(vocal), dtype = int)
    vfiction = np.zeros(len(vocal), dtype = int)
    vgovernment = np.zeros(len(vocal), dtype = int)
    vhumor = np.zeros(len(vocal), dtype = int)
    vnews = np.zeros(len(vocal), dtype = int)
    i = 0
    for (key) in vocal:
        vtext[i] = text_dic.get(key,0)
        vadventure[i] = adventure_dic.get(key,0)
        vfiction[i] = fiction_dic.get(key, 0)
        vgovernment[i] = government_dic.get(key, 0)
        vhumor[i] = humor_dic.get(key, 0)
        vnews[i] = news_dic.get(key, 0)
        i = i + 1

    print("adventure: ", np.dot(vtext,vadventure)/(np.linalg.norm(vtext)*np.linalg.norm(vadventure)))
    print("fiction: ", np.dot(vtext, vfiction) / (np.linalg.norm(vtext) * np.linalg.norm(vfiction)))
    print("government: ", np.dot(vtext, vgovernment) / (np.linalg.norm(vtext) * np.linalg.norm(vgovernment)))
    print("humor: ", np.dot(vtext, vhumor) / (np.linalg.norm(vtext) * np.linalg.norm(vhumor)))
    print("news: ", np.dot(vtext, vnews) / (np.linalg.norm(vtext) * np.linalg.norm(vnews)))

grail = read_text('data/grail.txt')
compare_to_brown(grail)
wsj = read_text('data/wsj')
compare_to_brown(wsj)

if __name__ == '__main__':

    text = read_text('data/grail.txt')
    token_count(text)
