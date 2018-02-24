#!/usr/bin/env python
from __future__ import division

import sys
logs = sys.stderr

# cat <kbest-list> | ./kbest_oracles.py <ref>

from deptree import DepTree, DepVal
from collections import defaultdict

import gflags as flags
FLAGS=flags.FLAGS

if __name__ == "__main__":

    # TODO: automatically figure out maxk
    flags.DEFINE_integer("maxk", 128, "maxk")

    try:
        file = open(sys.argv[1])
    except:
        print >> logs, "Usage: cat <kbest-lists> | ./kbest_oracles.py <goldtrees>"
        sys.exit(1)

    tot = defaultdict(lambda : DepVal())

    for sid, reftree in enumerate(DepTree.load(sys.argv[1]), 1):
        
        sentid, k = sys.stdin.readline().split()
        k = int(k)

        best = -1
        besttree = None
        for i in range(1, k+1):
            score, tree = sys.stdin.readline().split("\t")
            score = float(score)
            tree = DepTree.parse(tree)

            ev = reftree.evaluate(tree)
            if ev > best:
                best = ev
                besttree = tree

            tot[i] += best

        for i in range(k+1, FLAGS.maxk+1): # if short list
            tot[i] += best

        sys.stdin.readline()
        print "%s\t%s\t%s" % (sid, best, besttree)
        sys.stdout.flush()

    print 

    for k in [1,2,4,8,16,32,64,128]:
        if k in tot:
            print "%d\t%s\t%s" % (k, tot[k], tot[k].details())
