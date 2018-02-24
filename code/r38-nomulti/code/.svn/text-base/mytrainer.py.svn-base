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

import random # shuffle
import copy
from multiprocessing import cpu_count, Pool

class Perceptron(object):

    def __init__(self, decoder):

        self.trainfile = FLAGS.train
        self.trainlines = open(self.trainfile).readlines()

        self.devfile = FLAGS.dev
        self.devlines = open(self.devfile).readlines()

        self.outfile = FLAGS.out
        self.save_to = FLAGS.save_to
        self.decoder = decoder # a class, providing functions: load(), decode(), get_feats()
        self.iter = FLAGS.iter
        self.start_iter = FLAGS.start_iter
        self.avg = FLAGS.avg
        self.shuffle = FLAGS.shuffle

        self.weights = decoder.model.weights

        if FLAGS.resume_from is None:
            self.allweights = Vector()
            self.c = 0. # counter: how many examples have i seen so far? = it * |train| + i
        else:
            self.c = (self.start_iter - 1) * len(self.trainlines)
            self.allweights = Model(FLAGS.allweights).weights # read all weights from file

    def one_pass_on_train(self, data):
        
        num_updates = 0    
        early_updates = 0
##        for i, example in enumerate(self.decoder.load(self.trainfile), 1):
        for i, line in enumerate(data, 1):
            example = DepTree.parse(line)
            #print >> logs, "... example %d (len %d)..." % (i, len(example)),
            self.c += 1

            similarity, deltafeats = self.decoder.decode(example, early_stop=True)

            if similarity < 1 - 1e-8: #update

                num_updates += 1              

                #print >> logs, "sim={0}, |delta|={1}".format(similarity, len(deltafeats))
                if FLAGS.debuglevel >=2:
                    print >> logs, "deltafv=", deltafeats            
                
                self.weights += deltafeats
                if FLAGS.avg:
                    self.allweights += deltafeats * self.c

                if similarity < 1e-8: # early-update happened
                    early_updates += 1

            else:
                #print >> logs, "PASSED! :)"
                None

        return num_updates, early_updates

    def eval_on_dev(self):
        
        tot = self.decoder.evalclass()
        
##        for i, example in enumerate(self.decoder.load(self.devfile), 1):
        for i, example in enumerate(self.decoder.load(self.devlines), 1):
            similarity, _ = self.decoder.decode(example)
            tot += similarity
            
        return tot

    def dump(self, filename, weights):
        # calling model's write
        decoder.model.write(filename=filename, weights=weights)

    def avg_weights(self):
        return self.weights - self.allweights * (1/self.c)
        
    def train(self):

        if decoder.model.unk > 0:
            decoder.model.count_knowns_from_train(self.trainfile, self.devfile)
        
        starttime = time.time()        

        print >> logs, "starting perceptron at", time.ctime()

        best_prec = 0
        for it in xrange(self.start_iter, self.start_iter + self.iter):

            print >> logs, "iteration %d starts..............%s" % (it, time.ctime())

            iterstarttime = time.time()

            ncpus = 2#cpu_count()
            print >>logs, "Number of CPUs: %d"%ncpus
            lines = self.trainlines
            if self.shuffle:
                print >> logs, "Shuffling training set..."
                random.shuffle(lines)
                
            datas = [ [] for _ in range(ncpus)]

            for i,line in enumerate(lines):
                datas[i%ncpus].append(line)

            pool = Pool(processes=ncpus)
    
            l = pool.imap(worker, datas)
            pool.close()
            pool.join()
            num_updates, early_updates = 0, 0
            new_allweights, new_weights = None, None
            factor = 1.0/float(ncpus)
            for worker_result in l:
                my_num_updates, my_early_updates, my_allweights, my_weights = worker_result
                num_updates += my_num_updates
                early_updates += my_early_updates
                if self.avg:
                    if new_allweights==None:
                        new_allweights = my_allweights*factor
                    else:
                        new_allweights += my_allweights*factor
                if new_weights==None:
                    new_weights = my_weights*factor
                else:
                    new_weights += my_weights*factor

            self.c += len(lines)/ncpus
            self.allweights = new_allweights
            self.weights, self.decoder.model.weights = new_weights, new_weights

            print >> logs, "iteration %d training finished at %s. now evaluating on dev..." % (it, time.ctime())
            avgweights = self.avg_weights() if self.avg else self.weights
            if FLAGS.debuglevel >= 2:
                print >> logs, "avg w=", avgweights
            self.decoder.model.weights = avgweights
            prec = self.eval_on_dev()

            is_best_parsing = "+" if prec > best_prec else ""
            
            print >> logs, "at iteration %d, updates= %d (early %d), dev= %s%s, |w|= %d, time= %.3fh acctime= %.3fh" % \
                  (it, num_updates, early_updates, \
                   prec, is_best_parsing, \
                   len(avgweights), 
                   (time.time() - iterstarttime)/3600, (time.time() - starttime)/3600.)
            
            logs.flush()

            if prec > best_prec:
                best_prec = prec
                best_it = it
                best_wlen = len(avgweights)
                print >> logs, "new high at iteration {0}: {1}. Dumping Avg Weights...".format(it, prec)
                self.dump(self.outfile, avgweights)

            # to support resumed training, have to output non-avg weights
            if self.save_to is not None:
                print >> logs, "Dumping Non-Avg Weights...".format(it, prec)
                self.dump(self.save_to + ".nonavg", self.weights)
                print >> logs, "Dumping Add Weights...".format(it, prec)
                self.dump(self.save_to + ".all", self.allweights)

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

##    def load(self, filename):
    def load(self, lines, shuffle=False):

        if shuffle:
            print >> logs, "Shuffling training set..."
            random.shuffle(lines)            

##        for i, line in enumerate(open(filename), 1):
        for i, line in enumerate(lines, 1):
            yield DepTree.parse(line)

    def decode(self, reftree, early_stop=False):
        # TODO: train mode or test mode

        sentence = reftree.sent
        refseq = reftree.seq() # if early_stop else None # TODO: training with no earlystop

        mytree, myseq, score = self.parser.try_parse(sentence, refseq, early_stop=early_stop)
        if FLAGS.debuglevel >= 1:
            if early_stop:
                print >> logs, refseq 
                print >> logs, myseq
            else:            
                print >> logs, reftree
                print >> logs, mytree
            
        if early_stop: # train mode
            refseq = refseq[:len(myseq)] # truncate
            _, reffeats = self.parser.simulate(refseq, sentence) 
            _, myfeats = self.parser.simulate(myseq, sentence)  # memory consideration
            deltafeats = Model.trim(reffeats-myfeats)
        else: # test mode            
            deltafeats = None

        prec = reftree.evaluate(mytree)

        return prec, deltafeats


def worker(data):
    mytrainer = trainer
    num_updates, early_updates = mytrainer.one_pass_on_train(data)
    # print "mytrainer c: ", mytrainer.c, num_updates
    return num_updates, early_updates, mytrainer.allweights, mytrainer.weights
    
if __name__ == "__main__":

    flags.DEFINE_integer("iter", 1, "number of passes over the whole training data", short_name="i")
    flags.DEFINE_integer("start_iter", 1, "to resume for iteration...")
    flags.DEFINE_boolean("avg", True, "averaging parameters")
    flags.DEFINE_string ("train", None, "training corpus")
    flags.DEFINE_string ("dev", None, "dev corpus")
    flags.DEFINE_string ("out", None, "output file (for avg weights)")
    flags.DEFINE_string ("save_to", None, "output file (for non-avg and all weights) for resuming training")
    flags.DEFINE_string ("resume_from", None, "resume from FILE.nonavg and FILE.all")
    flags.DEFINE_string ("allweights", None, "all weights file (do not need to specify)")
    flags.DEFINE_boolean("shuffle", False, "randomize training data at each iteration")

    argv = FLAGS(sys.argv)
    FLAGS.svector = True # TODO: for now training has to use svector; will be removed
    
    if FLAGS.train is None or FLAGS.dev is None or FLAGS.out is None:
        print >> logs, "Error: must specify a training corpus, a dev corpus, and an output file (for weights)."  + str(FLAGS)
        sys.exit(1)

    if FLAGS.resume_from is not None:
        assert FLAGS.start_iter > 0
        FLAGS.allweights = FLAGS.resume_from + ".all" # read all weights from file
        FLAGS.weights = FLAGS.resume_from + ".nonavg"

    global decoder, trainer

    decoder = Decoder(Model(FLAGS.weights), b=FLAGS.b)
    trainer = Perceptron(decoder)

    trainer.train()
