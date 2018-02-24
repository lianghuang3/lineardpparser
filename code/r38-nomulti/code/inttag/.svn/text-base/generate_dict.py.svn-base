#!/usr/bin/env python

'''
Default:
    cat train.dep | ./generate_dict.py > train.dict
No char dict (for ENG):
    cat train.dep | ./generate_dict.py --cutoff_char 0 > train.dict
'''

from __future__ import division
import sys
logs = sys.stderr
from collections import defaultdict
from deptree import DepTree, DepVal
import gflags as flags
FLAGS=flags.FLAGS

def sorttags(tags):
    return "\t".join("%s %d" % (t, f) for (f,t) in sorted([(f, t) for (t, f) in tags.items()], reverse=True))

if __name__ == "__main__":

    flags.DEFINE_integer("cutoff", 1, "cut off freq")
    flags.DEFINE_integer("cutoff_char", 1, "cut off freq for chars")

    argv = FLAGS(sys.argv)

    word_dict = defaultdict(lambda: [0, defaultdict(int)])
    unktags = defaultdict(int)

    for line in sys.stdin:
        line = line.strip()
        if line == "":
            continue

        reftree = DepTree.parse(line)
        words = DepTree.words
        tags = reftree.tagseq()
        for w, t in zip(words, tags):
            word_dict[w][0] += 1
            word_dict[w][1][t] += 1

    unkcnt = 0
    for w in sorted(word_dict):
        freq, tags = word_dict[w]
        if freq <= FLAGS.cutoff:
            for t in tags:
                unktags[t] += tags[t]
                unkcnt += freq
        else:
            print "%s\t%d\t%s" % (w, freq, sorttags(tags))

    print "%s\t%d\t%s" % ("<unk>", unkcnt, sorttags(unktags))
        
#         freq_tags = str(words_tags[w][0]) + " " + " ".join(words_tags[w][1])
#         print w, freq_tags

    if FLAGS.cutoff_char > 0:
        print "------"

        char_dict = defaultdict(lambda: [0, defaultdict(int)])
        for w in sorted(word_dict):
            freq, tags = word_dict[w]
            if freq <= FLAGS.cutoff_char:
                for c in w.decode("utf8"):
                    c = c.encode("utf8")
                    char_dict[c][0] += freq
                    for t in tags:
                        char_dict[c][1][t] += tags[t]
        for c in sorted(char_dict):
            freq, tags = char_dict[c]
            print "%s\t%d\t%s" % (c, freq, sorttags(tags))

