#!/usr/bin/env python
from __future__ import division

import sys

# cat <kbest-list> | ./kbest_oracles.py <ref>

from deptree import DepTree, DepVal
from collections import defaultdict

if __name__ == "__main__":

    tot = defaultdict(lambda : DepVal())

    for sid, reftree in enumerate(DepTree.load(sys.argv[1]), 1):
        
        sentid, k = sys.stdin.readline().split()
        k = int(k)

        best = -1
        for i in range(1, k+1):
            score, tree = sys.stdin.readline().split("\t")
            score = float(score)
            tree = DepTree.parse(tree)

            ev = reftree.evaluate(tree)
            if ev > best:
                best = ev

            tot[i] += best

        sys.stdin.readline()
        print "%s\t%s" % (sid, best)

    for k in [1,2,4,8,16,32,64,128]:
        if k in tot:
            print "%d\t%s" % (k, tot[k])
