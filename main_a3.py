"""main_3a.py

An instance of the Text class should be initialized with a file path (a file or
directory). The example here uses object as the super class, but you may use
nltk.text.Text as the super class.

An instance of the Vocabulary class should be initialized with an instance of
Text (not nltk.text.Text).

"""

import os
import math

import nltk
from nltk.corpus import wordnet as wn
from nltk.probability import FreqDist
from nltk.text import Text as Text1
import re

# Part 1

STOPLIST = set(nltk.corpus.stopwords.words())
ENGLISH_VOCABULARY = set(w.lower() for w in nltk.corpus.words.words())

class Text:

    def __init__(self, path):
        self.path = path
        self.rawtext = self.read_text(path)
        self.nltktext = Text1(nltk.word_tokenize(self.rawtext))

    def __len__(self):
        return len(self.nltktext)

    def read_text(self, path):
        if os.path.isdir(path):
            files = []
            for i in os.listdir(path):
                file = open(os.path.join(path, i), 'r')
                files.append(file.read())
            files = '\n'.join(files)
            return files
        elif os.path.isfile(path):
            with open(path) as fh:
                return fh.read()

    def is_content_word(self, word):
        return word.lower() not in STOPLIST and word[0].isalpha()

    def token_count(self):
        return len(self.nltktext)

    def type_count(self):
        return len(set([w.lower() for w in self.nltktext]))

    def sentence_count(self):
        return len(nltk.sent_tokenize(self.rawtext))

    def most_frequent_content_words(self):
        dist = FreqDist([w for w in self.nltktext if self.is_content_word(w)])
        return dist.most_common(n=25)

    def most_frequent_bigrams(self):
        filtered_bigrams = [b for b in list(nltk.bigrams(self.nltktext))
                            if self.is_content_word(b[0]) and self.is_content_word(b[1])]
        dist = nltk.FreqDist([b for b in filtered_bigrams])
        return dist.most_common(n=25)

    def find_sirs(self):
        return list(sorted(set(re.findall(r'Sir\s[A-Z][a-z-]+', self.rawtext))))

    def find_brackets(self):
        return sorted(re.findall(r"\[.*\]|\(.*\)", self.rawtext))

    def find_roles(self):
        lis = list(set(re.findall(r"^.*(?=:)", self.rawtext, re.MULTILINE)))
        lis1 = []
        lis2 = []
        for w in lis:
            lis1.append(re.split(":", w)[0])
        for i in lis1:
            lis2.append(re.sub(r"SCENE.*", "-", i))
        return sorted(list(set([w for w in lis2 if w is not "-"])))

    def find_repeated_words(self):
        # return list(sorted(set(re.findall(r"\b(\w+)\b\s\b(\w+)\b\s\b(\w+)\b", self.rawtext))))
        return set([''.join(w) for w in re.findall(r'(\w{3,})(\s\1)(\2)', self.rawtext)])









class Vocabulary:

    def __init__(self, text):
        self.text = text
        self.all_items = set([w.lower() for w in text.nltktext])
        self.items = self.all_items.intersection(ENGLISH_VOCABULARY)
        # restricting the frequency dictionary to vocabulary items
        self.fdist = FreqDist(t.lower() for t in self.text.nltktext if t.lower() in self.items)
        self.text_size = len(self.text.nltktext)
        self.vocab_size = len(self.items)

    def __str__(self):
        return "<Vocabulary size=%d text_size=%d>" % (self.vocab_size, self.text_size)

    def __len__(self):
        return self.vocab_size

    def frequency(self, word):
        return self.fdist[word]

    def pos(self, word):
        if word not in self.items:
            return None
        synsets = wn.synsets(word)
        return synsets[0].pos() if synsets else 'n'

    def gloss(self, word):
        if word not in self.items:
            return None
        synsets = wn.synsets(word)
        return synsets[0].definition() if synsets else 'NO DEFINITION'

    def kwic(self, word):
        self.text.nltktext.concordance(word)

