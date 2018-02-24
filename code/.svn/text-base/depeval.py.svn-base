#!/usr/bin/env python
from __future__ import division

import sys
logs = sys.stderr

import gflags as flags
FLAGS=flags.FLAGS

from deptree import DepTree, DepVal

if __name__ == "__main__":

    flags.DEFINE_boolean("senteval", False, "sentence by sentence output", short_name="v")
    argv = FLAGS(sys.argv)
    if len(argv) != 3:
        print >> logs, "Usage: %s <file1> <file2>" % argv[0] + str(FLAGS)
        sys.exit(1)

    totalprec = DepVal()
    
    filea, fileb = open(argv[1]), open(argv[2])
    
    for i, (linea, lineb) in enumerate(zip(filea, fileb), 1):

        treea, treeb = map(DepTree.parse, (linea, lineb))
        prec = treea.evaluate(treeb)

        if FLAGS.senteval:
            print "sent {i:-4} (len {l}):\tprec= {p:.2%}".format(i=i, l=len(treea), p=prec.prec())

        totalprec += prec

    print "avg {a} sents,\tprec= {p:.2%}, {d:s}".format(a=i, p=totalprec.prec(), d=totalprec.details())
