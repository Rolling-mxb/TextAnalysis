import sys
import unittest
import warnings

from main_a4 import Text

def ignore_warnings(test_func):
    """Catching warnings via a decorator."""
    def do_test(self, *args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            test_func(self, *args, **kwargs)
    return do_test

class ExploreTextTests(unittest.TestCase):

    @classmethod
    @ignore_warnings
    def setUpClass(cls):
        cls.grail = Text('data/grail.txt')
        cls.nouns_more_common_in_plural = cls.grail.nouns_more_common_in_plural_form()
        cls.most_tags = cls.grail.which_word_has_greatest_number_of_distinct_tags()
        cls.frequent_tags = cls.grail.tags_in_order_of_decreasing_frequency()
        cls.fd = cls.grail.cfd

    def test_nouns_more_common_in_plural1(self):
        """To see whether the answer is empty or over lapping"""
        self.assertTrue(0 < len(self.nouns_more_common_in_plural) < 150)

    def test_nouns_more_common_in_plural2(self):
        """Make sure it has the right words"""
        nouns =['arthur', 'knights', 'halves', ',', 'snows', 'climes', 'strangers', 'ratios', 'wings', 'feathers',
                              'regulations', 'workers', 'differences', 'classes', 'people', 'affairs', 'angels', 'women', 'masses', '.']
        self.assertTrue(15 < len([t for t in self.nouns_more_common_in_plural if t in nouns]))

    def test_nouns_more_common_in_plural3(self):
        """Plural > Single"""
        self.assertTrue([self.fd[w] > self.fd[w[:-1]] for w in self.nouns_more_common_in_plural])

    def test_most_tags1(self):
        """check out frequency"""
        self.assertTrue(self.most_tags[0][1] > 5)

    def test_most_tags2(self):
        """The word with the most tags is 'arthur'."""
        self.assertEqual(self.most_tags[0][0], 'arthur')

    def test_most_tags3(self):
        """Make sure that top 5 on the list have at least 3 tags"""
        self.assertTrue(self.most_tags[5][1] > 3)

    def test_frequent_tags1(self):
        """The most frequent Brown tag is NN and it occurs 2284 times."""
        self.assertEqual(self.frequent_tags[0][0], 'NN')
        self.assertTrue(1000 < self.frequent_tags[0][1], 2284)

    def test_frequent_tags2(self):
        """Get the 15 most frequent tags and make sure the overlap is greater than 12."""
        most_frequent_tags = {'NN', 'NNP', '.', ':', 'PRP', 'DT', 'JJ', ',', 'IN', 'RB',
                              'VBP', 'VBZ', 'CC', 'VB', 'IN', 'NNS'}
        most_frequent_tags_found = set(t[0] for t in self.frequent_tags[:20])
        self.assertTrue(len(most_frequent_tags & most_frequent_tags_found) > 12)

    def test_frequent_tags3(self):
        """Make sure that top 5 tags on the list have appeared at least 800 times"""
        self.assertTrue(800 < self.frequent_tags[5][1])


if __name__ == '__main__':

    unittest.main()