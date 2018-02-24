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

flags.DEFINE_integer("iter", 1, "number of passes over the whole training data", short_name="i")
flags.DEFINE_boolean("avg", True, "averaging parameters")
flags.DEFINE_boolean("naiveavg", False, "naive sum for averaging (slow!)")
flags.DEFINE_string ("train", None, "training corpus")
flags.DEFINE_string ("dev", None, "dev corpus")
flags.DEFINE_string ("out", None, "output file (for weights)")
#flags.DEFINE_boolean("resort", False, "resort hashtable after each iter")
flags.DEFINE_integer ("multi", 1, "multiprocessing: # of CPUs, -1 to use all CPUs")

flags.DEFINE_boolean("singletrain", False, "use single thread for training")
flags.DEFINE_boolean("singledev", False, "use single thread for eval_on_dev")

flags.DEFINE_boolean("early", True, "use early update")

flags.DEFINE_boolean("shuffle", False, "shuffle training data before each iteraiton")

#flags.DEFINE_boolean("sortdelta", False, "averaging parameters")
from mydouble import counts

import gc
import os

class Perceptron(object):

    def __init__(self, decoder):

        Perceptron.early_stop = FLAGS.early        
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
        if Perceptron.avg:
            Perceptron.allweights = decoder.model.new_weights()
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

        # N.B.!
        Perceptron.decoder.parser.State.model.weights = Perceptron.weights # multiprocessing: State.model is static

        delta_weights, delta_allweights = self.decoder.model.new_weights(), self.decoder.model.new_weights()
        
        for i, example in enumerate(Perceptron.decoder.load(lines), 1):

            if Perceptron.ncpus == 1 and i % 1000 == 0:
                print >> logs, "... example %d (len %d)..." % (i, len(example)),

            similarity, deltafeats = Perceptron.decoder.decode(example, early_stop=Perceptron.early_stop, train=True)

            if similarity < 1 - 1e-8: #update

                num_updates += 1

                delta_len = len(deltafeats)
                
                if Perceptron.ncpus == 1 and i % 1000 == 0:
                    print >> logs, "sim={0}, |delta|={1}".format(similarity, delta_len)

                Perceptron.weights.iadd(deltafeats)
                
                delta_weights.iadd(deltafeats)

                if Perceptron.avg:
                    if FLAGS.naiveavg:
                        delta_allweights.iadd(Perceptron.weights) # naive sum
                    else:
                        delta_allweights.iaddc(deltafeats, Perceptron.c)
##                    Perceptron.allweights.iaddc(deltafeats, Perceptron.c)
                    
                if FLAGS.debuglevel >=2:
                    print >> logs, "deltafv=", type(deltafeats), deltafeats
                    print >> logs, "w=", Perceptron.weights
                    if Perceptron.avg:
                        print >> logs, "allw=", Perceptron.allweights

##                print >> logs, "|w|=%d |allw|=%d" % (len(Perceptron.weights), len(Perceptron.allweights))

                if similarity < 1e-8: # early-update happened
                    early_updates += 1

            else:
                if Perceptron.ncpus == 1 and i % 1000 == 0:
                    print >> logs, "PASSED! :)"

            Perceptron.c += 1

        return num_updates, early_updates, delta_weights, delta_allweights

    def avg_weights(self):
        if FLAGS.naiveavg:
##            return self.decoder.model.new_weights().get_avg(Perceptron.allweights, 1) # naive sum
            return Perceptron.allweights
        else:
            return Perceptron.weights.get_avg(Perceptron.allweights, Perceptron.c) # must not be self.c!

    def train_worker(self, sentences):
        tt = time.time()
        
        num_updates, early_updates, delta_weights, delta_allweights = self.one_pass_on_train(sentences)
        # print "mytrainer c: ", mytrainer.c, num_updates
        print >> logs, "inside a para time...", tt, time.time(), time.time() - tt
        
        return time.time() - tt, len(sentences), \
               (num_updates, early_updates), \
               (delta_weights, delta_allweights)
        
    def train(self):

        start_mem = memory()

        starttime = time.time()
##        model name	: Intel(R) Xeon(R) CPU           W3570  @ 3.20GHz

        print >> logs, "%d CPUs at %s %s" % (cpu_count(),
                                             os.popen("cat /proc/cpuinfo|grep GHz").readlines()[-1].strip().split(":")[-1],
                                             os.popen("cat /proc/cpuinfo|grep MHz").readlines()[-1].strip().split(":")[-1])
        
        print >> logs, "starting perceptron at", time.ctime()        

        best_prec = 0
        for it in xrange(1, self.iter+1):

            print >> logs, "iteration %d starts..............%s" % (it, time.ctime())

            curr_mem = memory() # outside of multi

            print >> logs, "memory usage at iter %d before pool: %s" % (it, human(memory(start_mem)))

            iterstarttime = time.time()

            if Perceptron.shuffle:
                self.shuffle_train()

            if not FLAGS.singletrain:
                pool = Pool(processes=self.ncpus)
            pool_time = time.time() - iterstarttime

            num_updates, early_updates = 0, 0
##            new_allweights, new_weights = self.decoder.model.new_weights(), self.decoder.model.new_weights()

            print >> logs, "memory usage at iter %d after pool: %s" % (it, human(memory(start_mem)))

            tt= time.time()
            print >> logs, "before para time...", tt
            results = map(self.train_worker, self.trainchunks) if FLAGS.singletrain else \
                          pool.map(self.train_worker, self.trainchunks, chunksize=1)

            if FLAGS.mydouble:
                print >> logs, "mydouble usage and freed: %d %d" % counts(), \
                      "|w|=", len(Perceptron.weights), "|avgw|=", len(Perceptron.allweights) if FLAGS.avg else 0, \
                      "|dw|=", len(results[0][-1][0])

            print >> logs, "after para time...", time.time()
            compute_time = time.time() - tt

            copy_time = 0
            para_times = []
            for dtime, size, (_num_updates, _early_updates), (_weights, _allweights) in results:

                num_updates += _num_updates
                early_updates += _early_updates

                factor = size / self.trainsize # not exactly uniform (if not equal-size split)!

                tt = time.time()
                if not FLAGS.singletrain:
                    Perceptron.weights.iaddc(_weights, factor)
#                 print _weights
#                 print new_weights
#                 print
                
                if self.avg:
                    if FLAGS.naiveavg:
                        Perceptron.allweights.iaddc(_allweights, factor)
                    else:
                        Perceptron.allweights.iaddc(_allweights, factor)

                del _weights, _allweights
                    
                copy_time += time.time() - tt

                para_times.append(dtime)

            del results
            
            if not FLAGS.singletrain:
                pool.close()
                pool.join()
##            else:
##                del self.delta_weights, self.delta_allweights # not in process
                
            print >> logs, "gc can't reach", gc.collect()

            print >> logs, "pool_time= %.1f s, compute_walltime= %.1f s, compute_cputime= %.1f (%s), copy_time= %.1f s" \
                  % (pool_time, compute_time, sum(para_times), " ".join("%.1f" % x for x in para_times), copy_time)
            
            print >> logs, "memory usage at iter %d after fork: %s" % (it, human(memory(start_mem)))            

            if not FLAGS.singletrain: # N.B.: in non-multiproc mode, self.c is updated
                Perceptron.c += self.trainsize / self.ncpus                
                print >> logs, "self.c=", Perceptron.c
            
#             if self.avg:
#                 Perceptron.allweights = new_allweights
#            Perceptron.weights, Decoder.model.weights = new_weights, new_weights
            
##            num_updates, early_updates = self.one_pass_on_train() # old single-cpu
            iterendtime = time.time()

            print >> logs, "memory usage at iter %d: extra %s, total %s" % (it,
                                                                            human(memory(curr_mem)),
                                                                            human(memory(start_mem)))
            if FLAGS.debuglevel >= 1:
                print >> logs, "weights=", Perceptron.weights

            curr_mem = memory()

                                                            
            print >> logs, "iteration %d training finished at %s. now evaluating on dev..." % (it, time.ctime())
            
            avgweights = self.avg_weights() if self.avg else Perceptron.weights
            if FLAGS.avg and FLAGS.debuglevel >= 1:
                print >> logs, "avgweights=", avgweights
                
            avgendtime = time.time()
            print >> logs, "avg weights (trim) took %.1f seconds." % (avgendtime - iterendtime)
            if FLAGS.debuglevel >= 2:
                print >> logs, "avg w=", avgweights
##            avgweights = self.decoder.model.new_weights()

            self.decoder.model.weights = avgweights # OK if noavg; see above
            Parser.State.model.weights = avgweights # multiprocessing: State.model is static

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

            self.decoder.model.weights = Perceptron.weights # restore non-avg

            del avgweights
            print >> logs, "gc can't reach", gc.collect()
            
            if FLAGS.mydouble:
                print >> logs, "mydouble usage and freed: %d %d ------------------------" % counts()

            logs.flush() # for hpc


        print >> logs, "peaked at iteration {0}: {1}, |bestw|= {2}.".format(best_it, best_prec, best_wlen)
        print >> logs, "perceptron training of %d iterations finished on %s (took %.2f hours)"  % \
              (it, time.ctime(), (time.time() - starttime)/3600.)

    def eval_worker(self, sentences):

        sub = decoder.evalclass() # global variable trainer

        for i, example in enumerate(self.decoder.load(sentences), 1):
##             tree = DepTree.parse(example) # have to parse here, not outside, because Tree.words is static
            similarity, _ = trainer.decoder.decode(example)
            sub += similarity # do it inside instead of outside

        return sub

    def eval_on_dev(self):        

        t = time.time()
        print >> logs, "garbage collection...", 
        gc.collect()
        print >> logs, "took %.1f seconds" % (time.time() - t)

        Parser.debuglevel = 0
        if FLAGS.multi is not None:
            print >>logs, "using %d CPUs for eval... chunksize=%d" % (self.ncpus, len(self.devchunks[0]))
            tot = self.decoder.evalclass()

            if not FLAGS.singledev:
                pool = Pool(processes=self.ncpus)

            for sub in map(self.eval_worker, self.devchunks) if FLAGS.singledev \
                    else pool.map(self.eval_worker, self.devchunks, chunksize=1):
                
                tot += sub

            if not FLAGS.singledev:
                pool.close()
                pool.join()
                
        else:
            tot = self.eval_worker(self.devlines)
            
        Parser.debuglevel = FLAGS.debuglevel # restore
        return tot

    def dump(self, weights):
        # calling model's write
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

    def decode(self, reftree, early_stop=False, train=True):

        sentence = reftree.sent

        refseq = reftree.seq() if train else None        

        mytree, myseq, score, _ = self.parser.try_parse(sentence, refseq, early_stop=early_stop, train=train)
        
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
