#!/usr/bin/env python

'''generic averaged perceptron trainer.'''

from __future__ import division

#####################################################

from multiprocessing import Pool
  
def _pickle_method(method):  
    func_name = method.im_func.__name__  
    obj = method.im_self  
    cls = method.im_class  
    return _unpickle_method, (func_name, obj, cls)  
  
def _unpickle_method(func_name, obj, cls):  
    for cls in cls.mro():  
        try:  
            func = cls.__dict__[func_name]  
        except KeyError:  
            pass  
        else:  
            break  
    return func.__get__(obj, cls)  
  
import copy_reg  
import types  
copy_reg.pickle(types.MethodType,  
    _pickle_method,  
    _unpickle_method)  

#####################################################

import math
import sys
logs = sys.stderr

import time
from mytime import Mytime
mytime = Mytime()

from model import Model
from deptree import DepTree, DepVal
from parser import Parser

from monitor import memory, human

import gflags as flags
FLAGS=flags.FLAGS

flags.DEFINE_integer("iter", 1, "number of passes over the whole training data", short_name="i")
flags.DEFINE_boolean("avg", True, "averaging parameters")
flags.DEFINE_string ("train", None, "training corpus")
flags.DEFINE_string ("dev", None, "dev corpus")
flags.DEFINE_string ("out", None, "output file (for weights)")
flags.DEFINE_boolean("resort", False, "resort hashtable after each iter")
flags.DEFINE_integer ("multi", 1, "multiprocessing: # of CPUs, -1 to use all CPUs")

#flags.DEFINE_boolean("sortdelta", False, "averaging parameters")

import gc

def me(x):
    return x+x

class Perceptron(object):

    def __init__(self, decoder):

        self.trainfile = FLAGS.train
        self.devfile = FLAGS.dev

        if FLAGS.multi != 1:
            t = time.time()
            print >> logs, "reading training / dev lines..."
            self.trainlines = open(self.trainfile).readlines()
            self.devlines = open(self.devfile).readlines()
            devsize = len(self.devlines)
            trainsize = len(self.trainlines)
            
            print >> logs, "%d training lines and %d dev lines read; took %.1f seconds" \
                  % (trainsize, devsize, time.time() - t)

            ncpus = FLAGS.multi #cpu_count()
            
            chunksize = int(math.ceil(devsize / ncpus))

            # split into roughly equally-sized chunks
            self.devchunks = []
            for j in range(0, devsize, chunksize):
                self.devchunks.append(self.devlines[j:j+chunksize]) # overflow OK

            chunksize = int(math.ceil(trainsize / ncpus))

            self.trainchunks = []
            for j in range(0, trainsize, chunksize):
                self.trainchunks.append(self.trainlines[j:j+chunksize]) # overflow OK
                
            print "dev: %d lines, %d chunks, %d per chunk" % (devsize, ncpus, len(self.devchunks[0]))
            print "train: %d lines, %d chunks, %d per chunk" % (trainsize, ncpus, len(self.trainchunks[0]))
        
        self.outfile = FLAGS.out
        self.decoder = decoder # a class, providing functions: load(), decode(), get_feats()
        self.iter = FLAGS.iter
        self.avg = FLAGS.avg

        self.weights = decoder.model.weights 
        if self.avg:
            self.allweights = decoder.model.new_weights()
        self.c = 0 # int! # counter: how many examples have i seen so far? = it * |train| + i

    def one_pass_on_train(self):
        
        num_updates = 0    
        early_updates = 0
        for i, example in enumerate(self.decoder.load(self.trainlines), 1):

            print >> logs, "... example %d (len %d)..." % (i, len(example)),

            similarity, deltafeats = self.decoder.decode(example, early_stop=True)

            if similarity < 1 - 1e-8: #update

                num_updates += 1

                delta_len = len(deltafeats)
                
                print >> logs, "sim={0}, |delta|={1}".format(similarity, delta_len)

                self.weights.iadd(deltafeats)
                if self.avg:
                    self.allweights.iaddc(deltafeats, self.c)
                    
                if FLAGS.debuglevel >=2:
                    print >> logs, "deltafv=", type(deltafeats), deltafeats
                    print >> logs, "w=", self.weights
                    print >> logs, "allw=", self.allweights

##                print >> logs, "|w|=%d |allw|=%d" % (len(self.weights), len(self.allweights))

                if similarity < 1e-8: # early-update happened
                    early_updates += 1

            else:
                print >> logs, "PASSED! :)"

            self.c += 1

        return num_updates, early_updates

    def eval_worker(self, sentences):

        sub = decoder.evalclass() # global variable trainer

        for i, example in enumerate(sentences):# trainer.decoder.load(sentences), 1):
            tree = DepTree.parse(example) # have to parse here, not outside, because Tree.words is static
            similarity, _ = trainer.decoder.decode(tree)
            sub += similarity # do it inside instead of outside

        return similarity

    def eval_on_dev(self):        

        t = time.time()
        print >> logs, "garbage collection...", 
        gc.collect()
        print >> logs, "took %.1f seconds" % (time.time() - t)

        Parser.debuglevel = 0
        if FLAGS.multi != 1:
            ncpus = FLAGS.multi
            print >>logs, "using %d CPUs for eval... chunksize=%d" % (ncpus, len(self.devchunks[0]))
            tot = self.decoder.evalclass()
               
            pool = Pool(processes=ncpus)

            for sub in pool.imap(self.eval_worker, self.devchunks, chunksize=1):
                tot += sub
        else:
            tot = self.eval_worker(self.devlines)
            
        Parser.debuglevel = FLAGS.debuglevel # restore
        return tot

    def dump(self, weights):
        # calling model's write
        decoder.model.write(filename=self.outfile, weights=weights)

    def avg_weights(self):
        return self.weights.get_avg(self.allweights, self.c)

    def train_worker(self, sentences):
        num_updates, early_updates = self.one_pass_on_train(sentences)
        # print "mytrainer c: ", mytrainer.c, num_updates
        return num_updates, early_updates, self.allweights, self.weights
        
    def train(self):

        start_mem = memory()

        starttime = time.time()

        print >> logs, "starting perceptron at", time.ctime()

        best_prec = 0
        for it in xrange(1, self.iter+1):

            print >> logs, "iteration %d starts..............%s" % (it, time.ctime())

            curr_mem = memory()
            iterstarttime = time.time()
            num_updates, early_updates = self.one_pass_on_train()
            iterendtime = time.time()

            print >> logs, "memory usage at iter %d: extra %s, total %s" % (it,
                                                                            human(memory(curr_mem)),
                                                                            human(memory(start_mem)))
            curr_mem = memory()

                                                            
            print >> logs, "iteration %d training finished at %s. now evaluating on dev..." % (it, time.ctime())
            if FLAGS.resort:
                t= time.time()
                self.weights = self.weights.resorted()
                self.allweights = self.allweights.resorted()
                print >> logs, "resort takes %.1f seconds." % (time.time() - t)
            
            avgweights = self.avg_weights() if self.avg else self.weights
            avgendtime = time.time()
            print >> logs, "avg weights (trim) took %.1f seconds." % (avgendtime - iterendtime)
            if FLAGS.debuglevel >= 2:
                print >> logs, "avg w=", avgweights
            self.decoder.model.weights = avgweights
            prec = self.eval_on_dev()
            print >> logs, "eval on dev took %.1f seconds." % (time.time() - avgendtime)

            
            print >> logs, "at iteration {0}, updates= {1} (early {4}), dev= {2}, |w|= {3}, time= {5:.3f}h acctime= {6:.3f}h"\
                  .format(it, num_updates, prec, len(avgweights), early_updates, \
                          (time.time() - iterstarttime)/3600, (time.time() - starttime)/3600.)
            logs.flush()

            if prec > best_prec:
                best_prec = prec
                best_it = it
                best_wlen = len(avgweights)
                print >> logs, "new high at iteration {0}: {1}. Dumping Weights...".format(it, prec)
                self.dump(avgweights)

            self.decoder.model.weights = self.weights # restore non-avg

            del avgweights
            gc.collect()
            
            if FLAGS.mydouble:
                from mydouble import counts
                print >> logs, "mydouble usage and freed: %d %d" % counts()


        print >> logs, "peaked at iteration {0}: {1}, |bestw|= {2}.".format(best_it, best_prec, best_wlen)
        print >> logs, "perceptron training of %d iterations finished on %s (took %.2f hours)"  % \
              (it, time.ctime(), (time.time() - starttime)/3600.)

class Decoder(object):

    '''providing three functions: load(), decode(early_stop=False), has attribute model, evalclass.'''

    def __init__(self, model, b=1):

        self.model = model
        self.b = b
        self.parser = Parser(model, self.b)
        self.evalclass = DepVal

    def load(self, lines):

        for i, line in enumerate(lines, 1):
            yield DepTree.parse(line)

    def decode(self, reftree, early_stop=False):

        sentence = reftree.sent

        refseq = reftree.seq() if early_stop else None        

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

            for i, (a, b) in enumerate(zip(refseq, myseq)):
                if a != b:
                    break            

            if True:
                _, deltafeats = self.parser.simulate(refseq, sentence, first_diff=i)
                _, deltafeats = self.parser.simulate(myseq, sentence, first_diff=i, actionfeats=deltafeats, c=-1)

                deltafeats.trim() #if FLAGS.svector else (reffeats, myfeats)
            else:
                _, reffeats = self.parser.simulate(refseq, sentence, first_diff=i)
                _, myfeats  = self.parser.simulate(myseq, sentence, first_diff=i)

                deltafeats = Model.trim(reffeats - myfeats) #if FLAGS.svector else (reffeats, myfeats)
                
        else: # test mode            
            deltafeats = None

        prec = reftree.evaluate(mytree)

        return prec, deltafeats

if __name__ == "__main__":

    flags.DEFINE_boolean("profile", False, "profile perceptron training")
    argv = FLAGS(sys.argv)
    
    if FLAGS.train is None or FLAGS.dev is None or FLAGS.out is None:
        print >> logs, "Error: must specify a training corpus, a dev corpus, and an output file (for weights)."  + str(FLAGS)
        sys.exit(1)

    global decoder
    decoder = Decoder(Model(FLAGS.weights), b=FLAGS.b)

    global trainer # used by pool workers
    trainer = Perceptron(decoder)

    if FLAGS.profile:
        import cProfile as profile
        profile.run('trainer.train()', '/tmp/a')
        import pstats
        p = pstats.Stats('/tmp/a')
        p.sort_stats('time').print_stats(50)

    else:
        trainer.train()
