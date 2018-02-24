#!/usr/bin/env python
from __future__ import division

import sys
logs = sys.stderr

import gflags as flags
FLAGS=flags.FLAGS
flags.DEFINE_boolean("verbatim", False, "sentence by sentence output", short_name="v")

from deptree import DepTree

if __name__ == "__main__":

    fileb = open(sys.argv[1])
    
    for i, (linea, lineb) in enumerate(zip(sys.stdin, fileb), 1):

        treea = DepTree.parse(linea)

        treea.replace_postags(map(lambda x:x.rsplit("/", 1)[1], lineb.split()))

        print treea
