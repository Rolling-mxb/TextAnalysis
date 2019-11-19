
import nltk
from nltk.corpus import brown


COMPILED_BROWN = 'brown.pickle'



class BrownCorpus(object):

    def __init__(self):
        self.cfd = nltk.ConditionalFreqDist((w.lower(), tag) for (w, tag) in brown.tagged_words())
        self.tagged_words = brown.tagged_words()
        self.pos_tags = [val for key, val in brown.tagged_words()]
        self.words = self.cfd.conditions()

def nouns_more_common_in_plural_form(bc):
    a = []
    for condition in bc.cfd.conditions():
        if bc.cfd[condition]['NNS'] > bc.cfd[condition[:-1]]['NN']:
            a.append(condition)
    return a

def which_word_has_greatest_number_of_distinct_tags(bc):
    num_tags = []
    for condition in bc.cfd.conditions():
        num_tags.append((condition, len(bc.cfd[condition])))
    return sorted(num_tags, key = lambda x: x[1],reverse=True)

def tags_in_order_of_decreasing_frequency(bc):
    fd = nltk.FreqDist(bc.pos_tags)
    return fd.most_common()

def tags_that_nouns_are_most_commonly_found_after(bc):
    tagbigram = nltk.bigrams(t for (w, t) in bc.tagged_words)
    afterN = nltk.FreqDist(t1 for (t1, t2) in tagbigram if t2.startswith('NN'))
    return afterN.most_common()

def proportion_ambiguous_word_types(bc):
    mono_tags = [condition for condition in bc.words if len(bc.cfd[condition]) > 1]
    return len(mono_tags) / len(bc.words)

def proportion_ambiguous_word_tokens(bc):
    a = [x[0].lower() for x in bc.tagged_words]
    mono_tags = [i for i in a if len(bc.cfd[i]) > 1]
    return len(mono_tags) / len(brown.words())
