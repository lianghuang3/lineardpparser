#!/usr/bin/env python

'''generic averaged perceptron trainer.'''

from __future__ import division

import math
import sys
logs = sys.stderr

import time
from mytime import Mytime

from model import Model
from deptree import DepTree, DepVal
from parser import Parser

from monitor import memory, human

import gflags as flags
FLAGS=flags.FLAGS

#flags.DEFINE_boolean("sortdelta", False, "averaging parameters")

import gc

class Perceptron(object):

    def __init__(self, decoder):

        self.trainfile = FLAGS.train
        self.devfile = FLAGS.dev
        self.outfile = FLAGS.out
        self.decoder = decoder # a class, providing functions: load(), decode(), get_feats()
        self.iter = FLAGS.iter
        self.avg = FLAGS.avg

        self.weights = decoder.model.weights 
        if self.avg:
            #self.allweights = decoder.model.new_weights()
            pass
        # next line is obsolete
        self.c = 0 # int! # counter: how many examples have i seen so far? = it * |train| + i

    def one_pass_on_train(self):
        
        num_updates = 0    
        early_updates = 0
        num_steps = 0
        for i, example in enumerate(self.decoder.load(self.trainfile), 1):

            if i % 1000 == 0:
                print >> logs, "... example %d (len %d)..." % (i, len(example)),

            similarity, deltafeats, steps = self.decoder.decode(example, early_stop=True)
            num_steps += steps

            if similarity < 1 - 1e-8: #update

                num_updates += 1

                delta_len = len(deltafeats)
                
                if i % 1000 == 0:
                    print >> logs, "sim={0}, |delta|={1}".format(similarity, delta_len)

                self.weights.iadd(deltafeats) # avg taken care of automatically
##                 if self.avg:
##                     self.allweights.iaddc(deltafeats, self.c)
                
                if FLAGS.debuglevel >=2:
                    print >> logs, "deltafv=", type(deltafeats), deltafeats
                    print >> logs, "w=", self.weights
##                     if self.avg:
##                         print >> logs, "allw=", self.allweights

##                print >> logs, "|w|=%d |allw|=%d" % (len(self.weights), len(self.allweights))

                if similarity < 1e-8: # early-update happened
                    early_updates += 1

            else:
                if i % 1000 == 0:
                    print >> logs, "PASSED! :)"

            self.weights.step() # c += 1
            self.c += 1

        return num_updates, early_updates, num_steps

    def eval_on_dev(self):

        t = time.time()
        print >> logs, "garbage collection...", 
        gc.collect()
        print >> logs, "took %.1f seconds" % (time.time() - t)
        
        tot = self.decoder.evalclass()

        Parser.debuglevel = 0

        for i, example in enumerate(self.decoder.load(self.devfile), 1):
            similarity, _, _ = self.decoder.decode(example)
            tot += similarity

        Parser.debuglevel = FLAGS.debuglevel
            
        return tot

    def dump(self, weights):
        # calling model's write
        decoder.model.write(filename=self.outfile, weights=weights)

    def avg_weights(self):
        # obsolete
        return self.weights.get_avg(self.allweights, self.c)
        
    def train(self):

        start_mem = memory()

        starttime = time.time()

        print >> logs, "starting perceptron at", time.ctime()

        best_prec = 0
        for it in xrange(1, self.iter+1):

            print >> logs, "iteration %d starts..............%s" % (it, time.ctime())

            curr_mem = memory()
            iterstarttime = time.time()
            self.decoder.num_edges = 0
            num_updates, early_updates, num_steps = self.one_pass_on_train()            
            iterendtime = time.time()


            print >> logs, "memory usage at iter %d: extra %s, total %s" % (it,
                                                                            human(memory(curr_mem)),
                                                                            human(memory(start_mem)))
            if FLAGS.debuglevel >= 1:
                print >> logs, "weights=", self.weights

            curr_mem = memory()

                                                            
            print >> logs, "iteration %d training finished at %s. now evaluating on dev..." % (it, time.ctime())
##            avgweights = self.avg_weights() if self.avg else self.weights

            avgtime = 0
            timer = Mytime()
            if self.avg:
##                print >> logs, "    w=", self.weights
##                print >> logs, " ".join(map(str, [x.get_step() for x in self.weights.values()]))
                self.weights.set_avg(self.c)
                avgtime += timer.gap()                
                if FLAGS.debuglevel >= 1:
                    print >> logs, "avgweights=", self.weights

            prec = self.eval_on_dev()
            
            print >> logs, "eval on dev took %.1f seconds." % timer.gap()

            print >> logs, "at iteration {0}, updates= {1} (early {4}), dev= {2}{7}, |w|= {3}, time= {5:.3f}h acctime= {6:.3f}h, root={10:.1%}"\
                  .format(it, num_updates, prec, len(self.weights), early_updates, 
                          (time.time() - iterstarttime)/3600,
                          (time.time() - starttime)/3600.,
                          "+" if prec > best_prec else "",
                          num_steps, self.decoder.num_edges,
                          prec.root())
            logs.flush()

            if prec > best_prec:
                best_prec = prec
                best_it = it
                best_wlen = len(self.weights)
                best_time = time.time() - starttime
                print >> logs, "new high at iteration {0}: {1}. Dumping Weights...".format(it, prec)
                if not FLAGS.dump_last:
                    self.dump(self.weights)
                else:
                    self.bestweights = self.weights.deepcopy()

            if self.avg:
                timer = Mytime()
                self.weights.reset_avg(self.c) # restore weights
                t = timer.gap()
                print >> logs, "avg weights (set/reset) took %.1f+%.1f=%.1f seconds." % (avgtime, t, avgtime+t)
            
##            self.decoder.model.weights = self.weights # restore non-avg

##            del avgweights
            gc.collect()
            
            if FLAGS.mydouble:
                from mydouble import counts
                print >> logs, "mydouble usage and freed: %d %d" % counts()


        print >> logs, "peaked at iteration {0}: {1} ({3:.1f}h), |bestw|= {2}.".format(best_it, best_prec,
                                                                                     best_wlen, best_time/3600)
        print >> logs, best_prec.details()
        print >> logs, "perceptron training of %d iterations finished on %s (took %.2f hours)"  % \
              (it, time.ctime(), (time.time() - starttime)/3600.)
        
        if FLAGS.dump_last:
            self.dump(self.bestweights)

class Decoder(object):

    '''providing three functions: load(), decode(early_stop=False), has attribute model, evalclass.'''

    def __init__(self, model, b=1):

        self.model = model
        self.b = b
        self.parser = Parser(model, self.b)
        self.evalclass = DepVal
        self.num_edges = 0

    def load(self, filename):

        for i, line in enumerate(open(filename), 1):
            yield DepTree.parse(line)

    def decode(self, reftree, early_stop=False):

        sentence = reftree.sent
        refseq = reftree.seq() if early_stop else None

        mytree, myseq, score, step = self.parser.try_parse(sentence, refseq, early_stop=early_stop)
        self.num_edges += self.parser.nedges
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

        return prec, deltafeats, step

    
if __name__ == "__main__":

    flags.DEFINE_boolean("profile", False, "profile perceptron training")
    flags.DEFINE_integer("iter", 1, "number of passes over the whole training data", short_name="i")
    flags.DEFINE_boolean("avg", True, "averaging parameters")
    flags.DEFINE_string ("train", None, "training corpus")
    flags.DEFINE_string ("dev", None, "dev corpus")
    flags.DEFINE_string ("out", None, "output file (for weights)")
    flags.DEFINE_float("learning_rate", 1.0, "learning rate")
    flags.DEFINE_boolean("dump_last", False, "dump best weights only at the end")

    argv = FLAGS(sys.argv)
    
    if FLAGS.train is None or FLAGS.dev is None: # or FLAGS.out is None:
        print >> logs, "Error: must specify a training corpus, a dev corpus."  + str(FLAGS)
        sys.exit(1)

    decoder = Decoder(Model(FLAGS.weights), b=FLAGS.b)

    trainer = Perceptron(decoder)

    if FLAGS.profile:
        import cProfile as profile
        profile.run('trainer.train()', '/tmp/a')
        import pstats
        p = pstats.Stats('/tmp/a')
        p.sort_stats('time').print_stats(50)

    else:
        trainer.train()
