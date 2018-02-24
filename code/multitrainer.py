#!/usr/bin/env python

'''generic averaged perceptron trainer.'''

from __future__ import division

#####################################################

from multiprocessing import Pool, cpu_count
  
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
import random

import gflags as flags
FLAGS=flags.FLAGS

flags.DEFINE_integer("iter", 15, "number of passes over the whole training data", short_name="i")
flags.DEFINE_boolean("avg", True, "averaging parameters")
flags.DEFINE_boolean("naiveavg", False, "naive sum for averaging (slow!)")
flags.DEFINE_string ("train", None, "training corpus")
flags.DEFINE_string ("dev", None, "dev corpus")
flags.DEFINE_string ("out", None, "output file (for weights)")
flags.DEFINE_boolean ("finaldump", False, "dump best weights at the end instead of at each new high")
flags.DEFINE_integer ("multi", 1, "multiprocessing: # of CPUs, -1 to use all CPUs")

##flags.DEFINE_boolean("singletrain", False, "use single thread for training")
##flags.DEFINE_boolean("singledev", False, "use single thread for eval_on_dev")
##flags.DEFINE_boolean("single", True, "--singletrain --singledev")

flags.DEFINE_string("update", "max", "update method: early, noearly, hybrid, late, max")

flags.DEFINE_boolean("shuffle", False, "shuffle training data before each iteration")

flags.DEFINE_float("learning_rate", 1.0, "learning rate")
flags.DEFINE_boolean("allmodels", False, "dumpto separate model files at each best-so-far iteration")

from mydouble import counts

import gc
import os

class Perceptron(object):

    def __init__(self, decoder):

        Perceptron.learning_rate = FLAGS.learning_rate
        Perceptron.singletrain = FLAGS.multi == 1
        Perceptron.singledev = FLAGS.multi == 1
        
        Perceptron.update = FLAGS.update

        Perceptron.trainfile = FLAGS.train
        Perceptron.devfile = FLAGS.dev
        Perceptron.outfile = FLAGS.out
        Perceptron.decoder = decoder # a class, providing functions: load(), decode(), get_feats()
        Perceptron.iter = FLAGS.iter
        Perceptron.avg = FLAGS.avg
        Perceptron.shuffle = FLAGS.shuffle
        
        if FLAGS.multi is not None:
            t = time.time()
            print >> logs, "reading training / dev lines..."
            Perceptron.trainlines = open(Perceptron.trainfile).readlines()
            Perceptron.trainsteps = sum((len(l.split()) * 2 - 1) for l in Perceptron.trainlines)
            print >> logs, "total steps", Perceptron.trainsteps
            Perceptron.devlines = open(Perceptron.devfile).readlines()
            Perceptron.devsize = len(Perceptron.devlines)
            Perceptron.trainsize = len(Perceptron.trainlines)
            
            print >> logs, "%d training lines and %d dev lines read; took %.1f seconds" \
                  % (Perceptron.trainsize, Perceptron.devsize, time.time() - t)

            Perceptron.ncpus = FLAGS.multi #cpu_count()
            
            chunksize = int(Perceptron.devsize / Perceptron.ncpus) # floor

            # split into roughly equally-sized chunks
            # why don't  i % ncpus? because i prefer consecutive sentences to be in the same chunk
            Perceptron.devchunks = []
            j = 0
            for i in range(Perceptron.ncpus):                
                Perceptron.devchunks.append(Perceptron.devlines[j:j+chunksize]) # overflow OK
                j += chunksize

            for k in range(j, Perceptron.devsize):    
                Perceptron.devchunks[k-j].append(Perceptron.devlines[k]) # first few chunks get slightly more

            self.shuffle_train()
                
            print "dev: %d lines, %d chunks, %d per chunk" % (Perceptron.devsize, Perceptron.ncpus, len(Perceptron.devchunks[0]))
            print "train: %d lines, %d chunks, %d per chunk" % (Perceptron.trainsize, Perceptron.ncpus, len(Perceptron.trainchunks[0]))

            #del Perceptron.trainlines, Perceptron.devlines
        
        Perceptron.weights = decoder.model.weights # must be static!
##         if Perceptron.avg:
##            Perceptron.allweights = decoder.model.new_weights()
        Perceptron.c = 0 # int! # counter: how many examples have i seen so far? = it * |train| + i

    def shuffle_train(self):
        
        if Perceptron.shuffle:
            random.shuffle(Perceptron.trainlines)
            print >> logs, "shuffling training data..."
            
        chunksize = int(math.ceil(Perceptron.trainsize / Perceptron.ncpus))
        Perceptron.trainchunks = []
        for j in range(0, Perceptron.trainsize, chunksize):
            Perceptron.trainchunks.append(Perceptron.trainlines[j:j+chunksize]) # overflow OK

    def one_pass_on_train(self, lines):

        num_updates = 0    
        early_updates = 0
        bad_updates = 0

        # N.B.!
##        print >> logs, "before this iteration, self.c =", Perceptron.c
        Perceptron.decoder.parser.State.model.weights = Perceptron.weights # multiprocessing: State.model is static

        delta_weights = self.decoder.model.new_weights()
##        delta_weights.set_step(Perceptron.weights.get_step()) # important!

        tot_steps = 0
        
        for i, example in enumerate(Perceptron.decoder.load(lines), 1):

            if Perceptron.ncpus == 1 and i % 1000 == 0:
                print >> logs, "... example %d (len %d)..." % (i, len(example)),

            similarity, deltafeats, steps = Perceptron.decoder.decode(example, train=True)
##            elif Perceptron.update == "hybrid":
##                similarity, deltafeats, steps = Perceptron.decoder.decode(example, early_stop=False, train=True)
##                viol = deltafeats.dot(Perceptron.weights)
##                if viol > 0: # bad update
##                    similarity, deltafeats, steps = Perceptron.decoder.decode(example, early_stop=True, train=True)  
##            elif Perceptron.update == "noearly":
##                similarity, deltafeats, steps = Perceptron.decoder.decode(example, early_stop=False, train=True)

            tot_steps += steps
            viol = deltafeats.dot(Perceptron.weights)
            if viol > 0: # bad update
                bad_updates += 1
                
            if similarity < 1 - 1e-8: #update

                num_updates += 1

                delta_len = len(deltafeats)
                
                if Perceptron.ncpus == 1 and i % 1000 == 0:
                    print >> logs, "sim={0}, |delta|={1}".format(similarity, delta_len)

                Perceptron.weights.iadd_wstep(deltafeats, Perceptron.c)
##                Perceptron.weights.set_avg(Perceptron.c)
##                print >> logs, "c=", Perceptron.c, "avgweights=", Perceptron.weights
##                Perceptron.weights.reset_avg(Perceptron.c)
                if not Perceptron.singletrain:
                    delta_weights.iadd_wstep(deltafeats, Perceptron.c)
                    
                if FLAGS.debuglevel >=2:
                    print >> logs, "deltafv=", type(deltafeats), deltafeats
                    print >> logs, "w=", Perceptron.weights
##                    if Perceptron.avg:
##                        print >> logs, "allw=", Perceptron.allweights

##                print >> logs, "|w|=%d |allw|=%d" % (len(Perceptron.weights), len(Perceptron.allweights))

                if steps < 2*len(example) - 1: # early-update happened
                    early_updates += 1

            else:
                if Perceptron.ncpus == 1 and i % 1000 == 0:
                    print >> logs, "PASSED! :)"

            Perceptron.c += 1

        return (num_updates, early_updates, tot_steps, bad_updates), delta_weights #, delta_allweights

    def train_worker(self, sentences):
        tt = time.time()
        
        update_stats, delta_weights = self.one_pass_on_train(sentences)
        # print "mytrainer c: ", mytrainer.c, num_updates
        print >> logs, "inside a para time...", tt, time.time(), time.time() - tt
        
        return time.time() - tt, len(sentences), \
               update_stats, \
               delta_weights #, delta_allweights)
        
    def train(self):

        start_mem = memory()       

        starttime = time.time()

        if FLAGS.finaldump:
            Perceptron.best_weights = self.decoder.model.new_weights()

##        model name	: Intel(R) Xeon(R) CPU           W3570  @ 3.20GHz

        print >> logs, "%d CPUs at %s %s" % (cpu_count(),
                                             os.popen("cat /proc/cpuinfo|grep [GM]Hz").readlines()[0].strip().split(":")[-1],
                                             os.popen("cat /proc/cpuinfo|grep [GM]Hz").readlines()[-1].strip().split(":")[-1])
        
        print >> logs, "starting perceptron at", time.ctime()        

        best_prec = 0
        acc_steps = 0
        for it in xrange(1, self.iter+1):
            Perceptron.curr = it 

            print >> logs, "iteration %d starts..............%s" % (it, time.ctime())

            curr_mem = memory() # outside of multi

            print >> logs, "memory usage at iter %d before pool: %s" % (it, human(memory(start_mem)))

            iterstarttime = time.time()

            if Perceptron.shuffle:
                self.shuffle_train()

            if not Perceptron.singletrain:
                pool = Pool(processes=self.ncpus)
            pool_time = time.time() - iterstarttime

            num_updates, early_updates, total_steps, bad_updates = 0, 0, 0, 0
##            new_allweights, new_weights = self.decoder.model.new_weights(), self.decoder.model.new_weights()

            print >> logs, "memory usage at iter %d after pool: %s" % (it, human(memory(start_mem)))

            tt= time.time()
            print >> logs, "before para time...", tt
            results = map(self.train_worker, self.trainchunks) if Perceptron.singletrain else \
                          pool.map(self.train_worker, self.trainchunks, chunksize=1)

            if FLAGS.mydouble:
                print >> logs, "mydouble usage and freed: %d %d" % counts(), \
                      "|w|=", len(Perceptron.weights), "|avgw|=", len(Perceptron.weights) if FLAGS.avg else 0, \
                      "|dw|=", len(results[0][-1])

            print >> logs, "after para time...", time.time()
            compute_time = time.time() - tt

            copy_time = 0
            para_times = []
            for dtime, size, (_num_updates, _early_updates, _steps, _bad_updates), _weights in results:

                num_updates += _num_updates
                early_updates += _early_updates
                total_steps += _steps
                bad_updates += _bad_updates
                
                factor = size / self.trainsize * Perceptron.learning_rate

                tt = time.time()
                if not Perceptron.singletrain: # singletrain: updated in place in one_pass_on_train()
                    Perceptron.weights.iaddc(_weights, factor)

                del _weights #, _allweights
                    
                copy_time += time.time() - tt

                para_times.append(dtime)

            del results
            
            if not Perceptron.singletrain:
                pool.close()
                pool.join()
                
            print >> logs, "gc can't reach", gc.collect()

            print >> logs, "pool_time= %.1f s, compute_walltime= %.1f s, compute_cputime= %.1f (%s), copy_time= %.1f s" \
                  % (pool_time, compute_time, sum(para_times), " ".join("%.1f" % x for x in para_times), copy_time)
            
            print >> logs, "memory usage at iter %d after fork: %s" % (it, human(memory(start_mem)))            

            if not Perceptron.singletrain: # N.B.: in non-multiproc mode, self.c is updated in place
                Perceptron.c += self.trainsize / self.ncpus

            print >> logs, "self.c=", Perceptron.c
            ##print >> logs, "w =", Perceptron.weights
            
            iterendtime = time.time()

            print >> logs, "memory usage at iter %d: extra %s, total %s" % (it,
                                                                            human(memory(curr_mem)),
                                                                            human(memory(start_mem)))
            if FLAGS.debuglevel >= 1:
                print >> logs, "weights=", Perceptron.weights

            curr_mem = memory()
                                                            
            print >> logs, "iteration %d training finished at %s. now evaluating on dev..." % (it, time.ctime())
            
##            avgweights = self.avg_weights() if self.avg else Perceptron.weights
            if self.avg:
##                Perceptron.weights.set_step(Perceptron.c)
                Perceptron.weights.set_avg(Perceptron.c)
                if FLAGS.debuglevel >= 1:
                    print >> logs, "avgweights=", self.weights
                
            avgendtime = time.time()
            print >> logs, "avg weights (trim) took %.1f seconds." % (avgendtime - iterendtime)
##            avgweights = self.decoder.model.new_weights()

            self.decoder.model.weights = Perceptron.weights # OK if noavg; see above
            Parser.State.model.weights = Perceptron.weights # multiprocessing: State.model is static

            prec = self.eval_on_dev()
            print >> logs, "eval on dev took %.1f seconds." % (time.time() - avgendtime)

            acc_steps += total_steps
            print >> logs, "at iter {0}, updates {1} (early {4}, er {10:.1f}%), dev {2}{7}, |w| {3}, time {5:.3f}h acctime {6:.3f}h; steps {8} cover {9:.1f}% accsteps {11}; bad {12} br {13:.1f}%"\
                  .format(it, num_updates, prec, len(Perceptron.weights), early_updates, \
                          (time.time() - iterstarttime)/3600,
                          (time.time() - starttime)/3600.,
                          "+" if prec > best_prec else "",
                          total_steps, 100.0*total_steps/Perceptron.trainsteps,
                          100.*early_updates/num_updates,
                          acc_steps,
                          bad_updates, 100.*bad_updates/num_updates) # 13 elements
            logs.flush()

            if prec > best_prec:
                best_prec = prec
                best_it = it
                best_wlen = len(Perceptron.weights)
                if not FLAGS.finaldump:
                    print >> logs, "new high at iteration {0}: {1}. Dumping Weights...".format(it, prec)
                    self.dump(Perceptron.weights, it)
                else:
                    Perceptron.best_weights = Perceptron.weights.copy()

            if self.avg:
                Perceptron.weights.reset_avg(Perceptron.c) # restore non-avg

            print >> logs, "gc can't reach", gc.collect()
            
            if FLAGS.mydouble:
                print >> logs, "mydouble usage and freed: %d %d ------------------------" % counts()

            logs.flush() # for hpc


        print >> logs, "peaked at iteration {0}: {1}, |bestw|= {2}.".format(best_it, best_prec, best_wlen)
        if FLAGS.finaldump:
            print >> logs, "Dumping best weights..."        
            self.dump(Perceptron.best_weights, best_it)
        print >> logs, "perceptron training of %d iterations finished on %s (took %.2f hours)"  % \
              (it, time.ctime(), (time.time() - starttime)/3600.)

    def eval_worker(self, sentences):

        sub = decoder.evalclass() # global variable trainer

        for i, example in enumerate(self.decoder.load(sentences), 1):
##             tree = DepTree.parse(example) # have to parse here, not outside, because Tree.words is static
            # N.B.: now use train instead of early_stop!
            similarity, _, _ = trainer.decoder.decode(example, train=False)
            sub += similarity # do it inside instead of outside

        return sub

    def eval_on_dev(self):        

        t = time.time()
        print >> logs, "garbage collection...", 
        gc.collect()
        print >> logs, "took %.1f seconds" % (time.time() - t)

        Parser.debuglevel = 0
        if not Perceptron.singledev and FLAGS.multi is not None:
            print >>logs, "using %d CPUs for eval... chunksize=%d" % (self.ncpus, len(self.devchunks[0]))
            tot = self.decoder.evalclass()

            if not Perceptron.singledev:
                pool = Pool(processes=self.ncpus)

            for sub in map(self.eval_worker, self.devchunks) if Perceptron.singledev \
                    else pool.map(self.eval_worker, self.devchunks, chunksize=1):
                
                tot += sub

            if not Perceptron.singledev:
                pool.close()
                pool.join()
                
        else:
            tot = self.eval_worker(self.devlines)
            
        Parser.debuglevel = FLAGS.debuglevel # restore
        return tot

    def dump(self, weights, it):
        # calling model's write
        if FLAGS.allmodels:
            decoder.model.write(filename=self.outfile+".i"+str(it), weights=weights)
        else:
            decoder.model.write(filename=self.outfile, weights=weights)

class Decoder(object):

    '''providing three functions: load(), decode(early_stop=False), has attribute model, evalclass.'''

    def __init__(self, model, b=1):

        Decoder.model = model
        Decoder.b = b
        Decoder.parser = Parser(model, Decoder.b)
        Decoder.evalclass = DepVal

    def load(self, lines):

        for i, line in enumerate(lines, 1):
            yield DepTree.parse(line)

    def decode(self, reftree, train=True):

        sentence = reftree.sent

        refseq = reftree.seq() if train else None        
        # (gold_feats, bad_feats) = \
        mytree, myseq, score, steps = self.parser.try_parse(sentence, refseq,
                                                            update=Perceptron.update if train else None)
        
        if FLAGS.debuglevel >= 1:
            if early_stop:
                print >> logs, refseq 
                print >> logs, myseq
            else:            
                print >> logs, reftree
                print >> logs, mytree
            
        if train: # train mode
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

        return prec, deltafeats, steps

if __name__ == "__main__":

    flags.DEFINE_boolean("profile", False, "profile perceptron training")
    argv = FLAGS(sys.argv)
    
    if FLAGS.train is None or FLAGS.dev is None:# or FLAGS.out is None:
        print >> logs, "Error: must specify a training corpus, a dev corpus."  + str(FLAGS)
        sys.exit(1)

    if FLAGS.update == "naive":
        FLAGS.naive = True # never merge for early update
#        FLAGS.multi = None

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
