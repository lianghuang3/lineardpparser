#!/usr/bin/env python

'''generic averaged perceptron trainer.'''

from __future__ import division

import math
import sys
logs = sys.stderr

from collections import defaultdict

from svector import Vector

import time
from mytime import Mytime
mytime = Mytime()

from model import Model
from deptree import DepTree, DepVal
from parser import Parser

import gflags as flags
FLAGS=flags.FLAGS

class Perceptron(object):

    def __init__(self, decoder):

        self.trainfile = FLAGS.train
        self.devfile = FLAGS.dev
        self.outfile = FLAGS.out
        self.decoder = decoder # a class, providing functions: load(), decode(), get_feats()
        self.iter = FLAGS.iter
        self.avg = FLAGS.avg

        self.weights = decoder.model.weights 
        self.allweights = Vector()
        self.c = 0. # counter: how many examples have i seen so far? = it * |train| + i

    def one_pass_on_train(self):
        
        num_updates = 0    
        early_updates = 0
        lenmyseqs = 0
        for i, example in enumerate(self.decoder.load(self.trainfile), 1):

            print >> logs, "... example %d (len %d)..." % (i, len(example)),

            self.c += 1

            similarity, deltafeats, lenmyseq = self.decoder.decode(example, early_stop=True)
            lenmyseqs += lenmyseq

            if similarity < 1 - 1e-8: #update

                num_updates += 1              

                print >> logs, "sim={0}, |delta|={1}".format(similarity, len(deltafeats))
                if FLAGS.debuglevel >=3:
                    print >> logs, "deltafv=", deltafeats            

                # lhuang
                assert "s0t=None=>LEFT" not in deltafeats, "%s %s" % (example.seq(), deltafeats)
                
                self.weights += deltafeats
                if FLAGS.avg:
                    self.allweights += deltafeats * self.c

                if similarity < 1e-8: # early-update happened
                    early_updates += 1

            else:
                print >> logs, "PASSED! :)"

        return num_updates, early_updates, lenmyseqs

    def eval_on_dev(self):
        
        tot = self.decoder.evalclass()
        
        for i, example in enumerate(self.decoder.load(self.devfile), 1):
            similarity, _, _ = self.decoder.decode(example)
            tot += similarity
            
        return tot

    def dump(self, weights):
        # calling model's write
        decoder.model.write(filename=self.outfile, weights=weights)

    def avg_weights(self):
        return self.weights - self.allweights * (1/self.c)
        
    def train(self):
        if decoder.model.unk > 0:
            decoder.model.count_knowns_from_train(self.trainfile, self.devfile)
        
        starttime = time.time()        

        print >> logs, "starting perceptron at", time.ctime()

        best_prec = 0
        best_tagprec = 0 # yang
        for it in xrange(1, self.iter+1):

            print >> logs, "iteration %d starts..............%s" % (it, time.ctime())

            iterstarttime = time.time()
            num_updates, early_updates, lenmyseq = self.one_pass_on_train()

            print >> logs, "iteration %d training finished at %s. now evaluating on dev..." % (it, time.ctime())
            avgweights = self.avg_weights() if self.avg else self.weights
            if FLAGS.debuglevel >= 2:
                print >> logs, "avg w=", avgweights
            self.decoder.model.weights = avgweights
            prec = self.eval_on_dev()

            # yang: output the logs for this iteration, tag prec added, "+" added
            best_parse, best_tag = "", ""
            if prec > best_prec:
                best_parse = "+"
            if prec.tagprec() > best_tagprec:
                best_tagprec = prec.tagprec()
                best_tag = "+"
            print >> logs, "at iteration {0}, updates= {1} (early {4}), dev= {2}{8}, |w|= {3}, time= {5:.3f}h acctime= {6:.3f}h, tagprec= {7:.2f}%{9}, unk= {11:.2f}%, knw= {12:.2f}%, |myseq|= {10}"\
                .format(it, num_updates, prec, len(avgweights), early_updates,
                            (time.time() - iterstarttime)/3600, (time.time() - starttime)/3600.,
                            prec.tagprec100(), best_parse, best_tag, lenmyseq,
                            prec.tagprec100_unk(), prec.tagprec100_knw()) # yang: tag prec added
            logs.flush()
            
            if prec > best_prec:
                best_prec = prec
                best_it = it
                best_wlen = len(avgweights)
                print >> logs, "new high at iteration {0}: {1}. Dumping Weights...".format(it, prec)
                self.dump(avgweights)

            self.decoder.model.weights = self.weights # restore non-avg

        if self.iter > 0: # iter==0 for some special debugging
            print >> logs, "peaked at iteration {0}: {1}, |bestw|= {2}.".format(best_it, best_prec, best_wlen)
            print >> logs, "perceptron training of %d iterations finished on %s (took %.2f hours)"  % \
              (it, time.ctime(), (time.time() - starttime)/3600.)

class Decoder(object):

    '''providing three functions: load(), decode(early_stop=False), has attribute model, evalclass.'''

    def __init__(self, model, b=1):

        self.model = model
        self.b = b
        self.parser = Parser(FLAGS.newstate, model, self.b)
        self.evalclass = DepVal

    def load(self, filename):

        for i, line in enumerate(open(filename), 1):
            yield DepTree.parse(line)

    def decode(self, reftree, early_stop=False):
        sentence = reftree.words # yang
        refseq = reftree.seq() # if early_stop else None # TODO: training with no earlystop
##        print >> logs, "REFSEQ", refseq

        gold_tags = reftree.tagseq() if FLAGS.use_gold_tags else None

        mytree, myseq, score = self.parser.try_parse(sentence, refseq, early_stop=early_stop, gold_tags=gold_tags)
##        print >> logs, "MYSEQ", myseq
        if FLAGS.debuglevel >= 1:
            if early_stop:
                print >> logs, refseq 
                print >> logs, myseq
            else:            
                print >> logs, reftree
                print >> logs, mytree
            
        if early_stop: # train mode
            if FLAGS.trunc_seq:
                refseq = Model.truncate_refseq(refseq, myseq)
            else:
                refseq = refseq[:len(myseq)] # truncate
            _, reffeats = self.parser.simulate(refseq, sentence) 
##            print >> logs, "\n\trefseq =", refseq 
##            print >> logs, "\n\t         ", _.tree()
            _, myfeats = self.parser.simulate(myseq, sentence) # memory consideration
##            print >> logs, "\n\t myseq =", myseq
##            print >> logs, "\n\t         ", _.tree()
##            print >> logs, "\n\treffeats:", len(reffeats), reffeats
##            print >> logs, "\n\t myfeats:", len(myfeats), myfeats

            deltafeats = Model.trim(reffeats-myfeats)
##            print >> logs, "\n\tdeltafeats:", len(deltafeats), deltafeats
        else: # test mode            
            deltafeats = None

        prec = reftree.evaluate(mytree)
##        print >> logs, reftree
##        print >> logs, mytree
##        print >> logs, prec

        return prec, deltafeats, len(myseq)

    
if __name__ == "__main__":

    flags.DEFINE_integer("iter", 1, "number of passes over the whole training data", short_name="i")
    flags.DEFINE_boolean("avg", True, "averaging parameters")
    flags.DEFINE_string ("train", None, "training corpus")
    flags.DEFINE_string ("dev", None, "dev corpus")
    flags.DEFINE_string ("out", None, "output file (for weights)")
    flags.DEFINE_boolean("trunc_seq", True, "truncate refseq according to tag action and parse action seperately")

    argv = FLAGS(sys.argv)
    FLAGS.svector = True # TODO: for now training has to use svector; will be removed
    
    assert FLAGS.shifttag + FLAGS.pretag + FLAGS.posttag == 1
    
    if FLAGS.train is None or FLAGS.dev is None or FLAGS.out is None:
        print >> logs, "Error: must specify a training corpus, a dev corpus, and an output file (for weights)."  + str(FLAGS)
        sys.exit(1)

    model = Model(FLAGS.weights)
    
    DepTree.model = model

    decoder = Decoder(model, b=FLAGS.b)

    trainer = Perceptron(decoder)

    trainer.train()
