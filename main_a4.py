import os
import nltk
from nltk.corpus import brown
import sys, getopt
from nltk.corpus import PlaintextCorpusReader
import pickle
import postag

class Text(object):

    def __init__(self, path, name=None):
        self.name = name
        if os.path.isfile(path):
            self.raw = open(path).read()
        elif os.path.isdir(path):
            corpus = PlaintextCorpusReader(path, '.*.mrg')
            self.raw = corpus.raw()
        self.text = nltk.text.Text(nltk.word_tokenize(self.raw))
        self.tagged_words = nltk.pos_tag(self.text)
        self.cfd = nltk.ConditionalFreqDist((w.lower(), tag) for (w, tag) in self.tagged_words)
        self.pos_tags = [val for key, val in self.tagged_words]
        self.words = self.cfd.conditions()



    def nouns_more_common_in_plural_form(self):
        a = []
        for condition in self.words:
            if self.cfd[condition]['NNS'] > self.cfd[condition[:-1]]['NN']:
                a.append(condition)
        return a

    def which_word_has_greatest_number_of_distinct_tags(self):
        num_tags = []
        for condition in self.cfd.conditions():
            num_tags.append((condition, len(self.cfd[condition])))
        return [w for w in sorted(num_tags, key=lambda x: x[1], reverse=True) if w[0].isalpha()]

    def tags_in_order_of_decreasing_frequency(self):
        fd = nltk.FreqDist(self.pos_tags)
        return fd.most_common()

    def tags_that_nouns_are_most_commonly_found_after(self):
        tagbigram = nltk.bigrams(t for (w, t) in self.tagged_words)
        afterN = nltk.FreqDist(t1 for (t1, t2) in tagbigram if t2.startswith('NN'))
        return afterN.most_common()

    def proportion_ambiguous_word_types(self):
        mono_tags = [condition for condition in self.words if len(self.cfd[condition]) > 1]
        return len(mono_tags) / len(self.words)

    def proportion_ambiguous_word_tokens(self):
        a = [x[0].lower() for x in self.tagged_words]
        mono_tags = [i for i in a if len(self.cfd[i]) > 1]
        return len(mono_tags) / len(self.text)


def main(argv):
    try:
        opts, args = getopt.getopt(argv[1:], 'h：', ["tagger-train", "tagger-run=", "tagger-test="])
    except getopt.GetoptError:
        print('get opt error')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print('["tagger-train","tagger-run=","tagger-test="]')
            sys.exit(2)
        elif opt == '--tagger-train':
            data = brown.tagged_sents(categories='news')
            tagger = postag.tagger_model_training(data)
            file_opened = open('model.txt', 'wb+')
            pickle.dump(tagger, file_opened)
            file_opened.close()
        elif opt == "--tagger-run":
            f = open('model.txt', 'rb+')
            tagger_model = pickle.load(f)
            f.close()
            test_sent = str(arg)
            tagged_sent = tagger_model.tag(nltk.word_tokenize(test_sent))
            sys.stdout.write(str(tagged_sent))
        elif opt == '--tagger-test':
            f = open('model.txt', 'rb+')
            tagger_model = pickle.load(f)
            f.close()
            test_dataset = postag.get_data(arg)
            sys.stdout.write(str(tagger_model.evaluate(test_dataset)))


if __name__ == "__main__":
    main(sys.argv)
