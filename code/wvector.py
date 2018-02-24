'''weight vector'''

from __future__ import division

#####################################################

# TODO: move to a wrapper of mydefaultdict

from _mycollections import mydefaultdict # tools
from collections import defaultdict

import __builtin__
__builtin__.mydefaultdict = mydefaultdict # pickle

from mydouble import mydouble, counts
__builtin__.mydouble = mydouble

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

##################

# TODO: do it in C
def _pickle_mydict(d):
##    print "hey, pickle, mydefaultdict!"
    # (key, (value, second)) - > (key, value, second)
    return _unpickle_mydict, (" ".join("%s=%s" % (x, "%s=%s" % y.pair()) for x, y in d.iteritems()),)

def take2(x):
    s, f, second = x.rsplit("=", 2) # (key, value, second)
    return s, mydouble(float(f), float(second))

def _unpickle_mydict(s):  
    return mydefaultdict(mydouble, (take2(x) for x in s.split()))
  
copy_reg.pickle(mydefaultdict,  
    _pickle_mydict,  
    _unpickle_mydict)  

######

def _pickle_dfdict(d):  
##    print "hey, pickle, defaultdict!"
    return _unpickle_dfdict, (d.items(),)  

def _unpickle_dfdict(s):  
    return defaultdict(mydefaultdict, s)
  
copy_reg.pickle(defaultdict,  
    _pickle_dfdict,  
    _unpickle_dfdict)

####

def _pickle_mydouble(d):
##    print "pickle mydouble",
    return _unpickle_mydouble, (float(d),)

def _unpickle_mydouble(d):
    return mydouble(float(d))


copy_reg.pickle(mydouble,  
    _pickle_mydouble,  
    _unpickle_mydouble)  

#####################################################

import copy # slow
import math

import sys
logs = sys.stderr

print >> logs, "using wvector..."

import gflags as flags
FLAGS=flags.FLAGS

flags.DEFINE_boolean("nonzerolen", False, "output non-zero feature len")
flags.DEFINE_boolean("trim", False, "trim")

flags.DEFINE_boolean("mydouble", True, "use C module mydouble instead of Python immutable float/int")

import gc
import time


##print >> logs, "using mydouble..."

class WVector(defaultdict):
    ''' wvector[action][feat] = weight'''

    action_names = None
    trim = False
    value_class = float # mydouble
    zero = 0

    @staticmethod
    def init(names):
        WVector.action_names = names
        WVector.dotrim = FLAGS.trim
        WVector.value_class = mydouble if FLAGS.mydouble else float
        WVector.zero = WVector.value_class(0) # mydouble(0)

    def __init__(self, value_class=None):
        if value_class is None:
            value_class = WVector.value_class

        # can add new actions on the fly; doesn't need to specify list of actions a priori
        # TODO: lambda : mydefaultdict(value_class)
        defaultdict.__init__(self, mydefaultdict,
                             [(action, mydefaultdict(value_class)) 
                              for action in WVector.action_names]) # doublehash 1

    def resorted(self):
        new = WVector()
        for action, feats in self.iteritems():
            new[action] = mydefaultdict(WVector.value_class, sorted(feats.items()))
                
        del self
        return new
        
    def evaluate(self, action, list_of_features):
        return self[action].evaluate(list_of_features)
    
#         w = self[action]
#         return sum(map(w.__getitem__, list_of_features))
        
    def iadd(self, other):

        for (action, feats) in other.iteritems():
            self[action].iadd(feats) # int defaultdict
            
        return self

    def iadd_wstep(self, other, step=1):

        for (action, feats) in other.iteritems():
            self[action].iadd(feats, step) # int defaultdict
            
        return self

    def iaddc(self, other, c=1):

        for (action, feats) in other.iteritems():
            self[action].iaddc(feats, c)
                
        return self

    def iaddl(self, action, other, c=1):
        self[action].iaddl(other, c)

    def dot(self, other):
        '''dot product'''
        s = 0
        for (action, feats) in other.iteritems():
            s += self[action].dot(feats)
        return s

##    def iadd2(self, other, other2, c2):
##        ''' c=1, c2=int '''

##        for (action, feats) in other.iteritems():
##            my = self[action]
##            for feat, value in feats.iteritems():
##                if WVector.dotrim and my[feat] == -value:                    
##                    del my[feat]
##                else:
##                    my[feat] += value

##            if other2 is not None and c2 != 0:
##                my2 = other2[action]
##                for feat, value in feats.iteritems():
##                    v = value * c2
##                    if WVector.dotrim and my2[feat] == -v:
##                        del my2[feat]
##                    else:
##                        my2[feat] += v # c2 is int
                            
##        return self

##    def iadd2(self, (plus, minus), c=1, other=None, c2=1):
##        for action in WVector.action_names:
##            my = self[action]

##            plusfeats = sorted(plus[action]) + [None]
##            minusfeats = sorted(minus[action]) + [None]
####            print >> logs, action, plusfeats
####            print >> logs, action, minusfeats

##            i = j = 0
##            fa = plusiter.next()
##            fb = minusiter.next()
                    
##            while fa is not None and fb is not None:
##                if fa == fb:
##                    fa = plusiter.next()
##                    fb = minusiter.next()
##                elif fa < fb:
##                    my[fa] += c                    
##                    fa = plusiter.next()
##                else:
##                    my[fb] -= c
##                    fb = minusiter.next()
                    
##            for f in plusiter[i:-1]:
##                my[f] += c

##            for f in minusfeats[j:-1]:
##                my[f] -= c
                
##        return self        

    def get_avg(self, all_weights, c):
        ''' return self.weights - self.allweights * (1/self.c) '''

##        return self.copy().iaddc(all_weights, -1./c)
        new = WVector()
        for action, feats in self.iteritems():
            new[action] = feats.addc(all_weights[action], -1./c) # now mydouble is mutable, must deepcopy

        return new
    
        return w.trim() if FLAGS.trim else w

    def step(self, s=1):
        for _, feats in self.iteritems():
            feats.step(s)

    def get_step(self):
        for _, feats in self.iteritems():
            return feats.get_step()

    def set_step(self, s=0):
        for _, feats in self.iteritems():
            return feats.set_step(s)

    def set_avg(self, c):
        for _, feats in self.iteritems():
            feats.set_avg(c)

    def reset_avg(self, c):
        for _, feats in self.iteritems():
            feats.reset_avg(c)

    def copy(self):
        ''' should be made a lot faster!!'''

        t = time.time()
        new = WVector()
        for action, feats in self.iteritems():
            new[action] = copy.deepcopy(feats) # now mydouble is mutable, must deepcopy

        print >> logs, "copying took %.1f seconds" % (time.time() - t)
        return new

    def deepcopy(self):
        new = WVector()
        for action, feats in self.iteritems():
            new[action] = feats.deepcopy()

    def get_flat_weights(self):
        '''return single-layer dictionary'''
        w = {}
        for action, feats in self.iteritems():
            for f, v in feats.iteritems():
                w["%s=>%s" % (f, action)] = v
        return w

    def trim(self):
        for feats in self.itervalues():
            for f, v in feats.items():
                if v == 0 or v == WVector.zero:
                    #del v # N.B. free this mydouble instance!
                    del feats[f]

##        t = time.time()
##        for action, feats in self.items():
##            new = defaultdict(int)
##            for f, v in feats.items():
##                if math.fabs(v) > 1e-3:
##                    new[f] = v
##            self[action] = new

##        print >> logs, "trimming took %.1f seconds" % (time.time() - t)

##        gc.collect()
        
##        return self

    def __len__(self):
        ''' non-zero length '''
        return sum(map(len, self.values())) if not FLAGS.nonzerolen else \
               sum(len(filter(lambda v:math.fabs(v) > 1e-3, feats.values())) \
                   for feats in self.values())

    def __str__(self):
        s = []
        for action, feats in self.iteritems():
            for f, v in feats.items():
##                if math.fabs(v) > 1e-3:
                s.append("%s=>%s=%s" % (f, action, v))
        return " ".join(s)

#     def evaluate(self, feats):
#         '''like dot-product, but feats is a list of features'''
#         return sum(map(self.__getitem__, feats)) # map is fast
