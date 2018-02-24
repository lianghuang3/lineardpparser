'''vocabulary'''

from collections import defaultdict

import gflags as flags
FLAGS=flags.FLAGS

flags.DEFINE_boolean("integerize", False, "(s0t, s1t) instead of \"s0t-s1t\"")

class Vocab(object):

    _words = {} # defaultdict(lambda: len(Vocab._words))
    _index = {}

    @staticmethod
    def str2id(word):
        i = Vocab._words.get(word, None)
        if i is None:
            i = len(Vocab._words)
            Vocab._words[word] = i
            Vocab._index[i] = word
        return i            

    @staticmethod
    def id2str(id):
        return Vocab._index[id]
