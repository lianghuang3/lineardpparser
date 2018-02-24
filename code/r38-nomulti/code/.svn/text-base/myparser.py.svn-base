#!/usr/bin/env python
from __future__ import division

from multiprocessing import cpu_count, Pool

import sys

import readline
def shell_input():
    try:
        while True:
            yield raw_input()
    except:
        return

logs = sys.stderr

from collections import defaultdict

from svector import Vector
from model import Model
from deptree import DepTree, DepVal

from mytime import Mytime
mytime = Mytime()

import gflags as flags
FLAGS=flags.FLAGS

flags.DEFINE_integer("beam", 1, "beam width", short_name="b")
flags.DEFINE_integer("leftbeam", 1000, "leftptrs beam width") # number of left items (predictors to be combined w/ current)
flags.DEFINE_integer("kbest", 0, "kbest", short_name="k")
flags.DEFINE_boolean("forest", False, "dump the forest")
flags.DEFINE_boolean("earlystop", False, "try early stop (compared with gold seq)")
flags.DEFINE_integer("debuglevel", 0, "debug level (0: no debug info, 1: brief, 2: detailed)", short_name="D")
flags.DEFINE_boolean("dp", False, "use dynamic programming (merging)")
flags.DEFINE_boolean("uniqstat", False, "print uniq states stat info")
flags.DEFINE_boolean("seq", False, "print action sequence")
flags.DEFINE_string("sim", None, "simulate action sequences from FILE", short_name="s")

flags.DEFINE_boolean("profile", False, "profile")

flags.DEFINE_boolean("newstate", True, "stackless treeless state")

flags.DEFINE_boolean("new", False, "new dp beaming")
flags.DEFINE_boolean("oracle", False, "forest oracle")

#flags.DEFINE_boolean("donotcarei", False, "left boundary i not included in signature (state equality)")

#TODO: feature reengineering + learning
#TODO: cube pruning
#TODO: forest generation (and k-best)

uniqstats = defaultdict(list)

class Parser(object):

    State = None

    def __init__(self, newstate=True, model=None, b=FLAGS.b):
        
        self.model = model
        self.b = b
        self.State = __import__("newstate").NewState
        self.State.model = model

    def try_parse(self, sent, refseq=None, early_stop=False):
        ''' returns myseq, mytree, goodfeats, badfeats'''

        self.State.setup()
        self.State.sent = sent
        
        n = len(sent)

        beams = [None] * (2*n)
        beams[0] = [self.State.initstate()] # initial state

        self.nstates = 0  # space complexity
        self.nedges = 0 # time complexity
        self.nuniq = 0
        if FLAGS.debuglevel >=2 and early_stop:
            print >> logs, "gold positions: ",
        for i in range(1, 2*n): # 2n-1 steps

            buf = []
            gold_item = None
            for old in beams[i-1]:
                for action in old.allowed_actions():
                    action_gold = refseq is not None and action == refseq[i-1]  # 0-based
                    for new in old.take(action, action_gold):
                        buf.append(new)
                        if early_stop and new.gold:
                            gold_item = new

            self.nedges += len(buf)

            if FLAGS.debuglevel >= 2:
                print >> logs, "\n".join([x.__str__(top=True) for x in sorted(buf)]) # beams[i]])
                print >> logs

            buf = sorted(buf) #[:self.b]
            tmp = {}
            beams[i] = []
            for j, new in enumerate(buf):
                if not FLAGS.dp or new not in tmp:
                    tmp[new] = new  ## first time to eval feats
                    new.rank = len(beams[i])
                    beams[i].append(new)
                else:
                    tmp[new].mergewith(new)
                    if FLAGS.debuglevel >=1 and early_stop and new.gold:
                        print >> logs, "GOLD at %d MERGED with %d! " % (j, tmp[new].rank),

                if not FLAGS.new:
                    if j == self.b - 1:  # |non-uniq|=b
                        break
                elif len(tmp) == self.b: # |uniq|=b
                    break                    

            self.nstates += len(beams[i])
            self.nuniq += len(tmp)            
            uniqstats[i].append(len(tmp)) #global variable

            if FLAGS.debuglevel >= 2:
                print >> logs, "\n".join([x.__str__(top=True) for x in beams[i]])
                print >> logs

            if early_stop:
                if gold_item.rank == -1: # early stop
                    if FLAGS.debuglevel >=2: 
                        print >> logs
                    if FLAGS.debuglevel >=1: 
                        print >> logs, "failed at step %d (len %d): %s" % (i, n, beams[i][0])
                    return None, beams[i][0].all_actions(), 0  ## no tree, no score
                else:
                    if FLAGS.debuglevel >=2:                                 
                        print >> logs, "-%d-" % gold_item.rank,
        
        if FLAGS.debuglevel >=1 and early_stop:
            print >> logs # after gold positions line
            if gold_item.rank == 0:
                print >> logs, "succeeded at step %d" % i
            else:
                print >> logs, "finished wrong at step %d" % i            

        goal = beams[-1][0]
        self.beams = beams

        return goal.tree(), goal.all_actions(), goal.score

    def stats(self):
        return (self.nstates, self.nedges, self.nuniq)

    def simulate(self, actions, sent):
        '''simulate the result of a given sequence of actions'''

        self.State.sent = sent

        n = len(sent)
        state = self.State.initstate() # initial state
        actionfeats = Vector()

        for i, action in enumerate(actions, 1):

##            actionfeats += state.make_feats(action) ## has to be OLD STATE -- WHY?
            for feat in state.make_feats(action):
                actionfeats[feat] += 1
            
            if action in state.allowed_actions():
                for new in state.take(action):
                    state = new
                    break
            else:
                print >> logs, "Error! BAD SEQUENCE!"
                break

        return state, actionfeats

    def dumpforest(self, id=0):
        print "sent.%d\t%s" % (id, " ".join(["%s/%s" % wt for wt in self.State.sent]))
        finalbeam = self.beams[-1]
        nodes = set()
        nodeids = set()
        for x in finalbeam:
            x.previsit(nodes, nodeids)
            
##        print sum([len(beam) for beam in self.beams])
        print len(nodes) + 1 # number of nodes

        cache = set()
        for x in finalbeam:
            x.postvisit(cache)        

        # final root node
        print "%d\t-1 [%d-%d]\t%d ||| " % (len(nodes)+1, 0, len(self.State.sent), len(finalbeam))
        for state in finalbeam:
            print "\t%d ||| 0=0" % state.nodeid

        print

    def forestoracle(self, reftree):

        reflinks = reftree.links()
        oracle = 0

        for i, state in enumerate(self.beams[-1]):
            h = state.headidx
            root = 1 if (h in reflinks and reflinks[h] == -1) else 0
            rooteval = DepVal(yes=root, tot=1) # root link

            subeval, tree = state.besteval(reflinks)
            
            if rooteval + subeval > oracle:
#                print i, rooteval, subeval
                oracle = rooteval + subeval
                oracletree = tree

        print >> logs, "oracle=", oracle, reftree.evaluate(oracletree)
#        print "oracle=", oracletree
        return oracle, oracletree
            

####################################################################

def work(line, i, parser):
    
    # global totalscore, totalstates, totaledges, totaluniq, totaltime, totalprec

    line = line.strip()
    if line[0]=="(":
        # input is a gold tree (so that we can evaluate)
        reftree = DepTree.parse(line)
        sentence = DepTree.sent # assigned in DepTree.parse()            
    else:
        # input is word/tag list
        reftree = None
        sentence = [tuple(x.rsplit("/", 1)) for x in line.split()]   # split by default returns list            
        DepTree.sent = sentence

    if FLAGS.debuglevel >= 1:
        print >> logs, sentence
        print >> logs, reftree

    mytime.zero()

    if FLAGS.sim is not None: # simulation, not parsing
        actions = map(int, sequencefile.readline().split())
        goal, feats = parser.simulate(actions, sentence) #if model is None score=0
        print >> logs, feats
        score, tree = goal.score, goal.top()
        (nstates, nedges, nuniq) = (0, 0, 0)
    else:
        # real parsing
        if True: #FLAGS.earlystop:
            refseq = reftree.seq() if reftree is not None else None
            tree, myseq, score = parser.try_parse(sentence, refseq, early_stop=FLAGS.earlystop)
            if FLAGS.earlystop:
                print >> logs, "ref=", refseq
                print >> logs, "myt=", myseq

                refseq = refseq[:len(myseq)] # truncate
                _, reffeats = parser.simulate(refseq, sentence) 
                _, myfeats = parser.simulate(myseq, sentence)
                print >> logs, "feat diff=", Model.trim(reffeats-myfeats)


            nstates, nedges, nuniq = parser.stats()
        else:
            goal = parser.parse(sentence)
            nstates, nedges, nuniq = parser.stats()

    dtime = mytime.period()

    if not FLAGS.earlystop and not FLAGS.profile:
        if FLAGS.forest:
            parser.dumpforest(i)
        else:
            if not FLAGS.kbest:
                toprint = str(tree)
            else:
                stuff = parser.beams[-1][:FLAGS.kbest]
                toprint = "sent.%d\t%d" % (i, len(stuff))
                toprint += ["%.2f\t%s" % (state.score, state.tree()) for state in stuff]

        if FLAGS.oracle:
            oracle, oracletree = parser.forestoracle(reftree)
            totaloracle += oracle

    prec = DepTree.compare(tree, reftree) # OK if either is None

    searched = sum(x.derivation_count() for x in parser.beams[-1]) if FLAGS.forest else 0
    print >> logs, "sent {i:-4} (len {l}):\tmodelcost= {c:.2f}\tprec= {p:.2%}"\
          "\tstates= {ns} (uniq {uq})\tedges= {ne}\ttime= {t:.3f}\tsearched= {sp}" \
          .format(i=i, l=len(sentence), c=score, p=prec.prec(), \
                  ns=nstates, uq=nuniq, ne=nedges, t=dtime, sp=searched)
    if FLAGS.seq:
        actions = goal.all_actions()
        print >> logs, " ".join(actions)
        check = simulate(actions, sentence, model) #if model is None score=0
        checkscore = check.score
        checktree = check.top()
        print >> logs, checktree
        checkprec = checktree.evaluate(reftree)
        print >> logs, "verify: tree:%s\tscore:%s\tprec:%s" % \
              (tree == checktree, score == checkscore, prec == checkprec)
        print >> logs, "sentence %-4d (len %d): modelcost= %.2lf\tprec= %.2lf\tstates= %d (uniq %d)\tedges= %d\ttime= %.3lf" % \
              (i, len(sentence), checkscore, checkprec.prec100(), nstates, nuniq, nedges, dtime)

    return toprint,  (score, nstates, nedges, nuniq, dtime, prec)

def worker_process(data):

    totalscore = 0
    totalstates = 0
    totaluniq = 0
    totaledges = 0
    totaltime = 0
    totalprec = DepVal()

    for line, i in data:
        parse_result, (score, nstates, nedges, nuniq, dtime, prec) = work(line,i,parser) # add parser here improve the performance, maybe namespace searching issue?
        totalscore += score
        totalstates += nstates
        totaledges += nedges
        totaluniq += nuniq
        totaltime += dtime
        totalprec += prec
    return totalscore, totalstates, totaledges, totaluniq, totaltime, totalprec

def main():

    if FLAGS.sim is not None:
        sequencefile = open(FLAGS.sim)

    if FLAGS.weights is None:
        if not FLAGS.sim:
            print >> logs, "Error: must specify a weights file" + str(FLAGS)
            sys.exit(1)
        else:
            model = None # can simulate w/o a model
    else:
        model = Model(FLAGS.weights) #FLAGS.model, FLAGS.weights)
    
    print >> logs, "knowns", len(model.knowns)

    # global totalscore, totalstates, totaledges, totaluniq, totaltime, totalprec
    
    totalscore = 0
    totalstates = 0
    totaluniq = 0
    totaledges = 0
    totaltime = 0
    totalprec = DepVal()
    
    totaloracle = DepVal()    

    global parser
    parser = Parser(FLAGS.newstate, model, b=FLAGS.beam)

    ncpus = cpu_count()
        
    datas = [ [] for i in range(ncpus)]
    for i, line in enumerate(sys.stdin, 1):
        datas[i%ncpus].append( (line,i) )
    
    print >> logs, "using %d CPUs" % ncpus
    pool = Pool(processes=ncpus)
    
    l = pool.imap(worker_process, datas)
    pool.close()
    pool.join()
    #exit()
    for x in l:
        (score, nstates, nedges, nuniq, dtime, prec) = x
        totalscore += score
        totalstates += nstates
        totaledges += nedges
        totaluniq += nuniq
        totaltime += dtime
        totalprec += prec

    #if FLAGS.newstate:
        #print >> logs, "feature constructions: tot= %d shared= %d (%.2f%%)" % (State.tot, State.shared, State.shared / State.tot * 100)

    print >> logs, "beam= {b}, avg {a} sents,\tmodelcost= {c:.2f}\tprec= {p:.2%}" \
          "\tstates= {ns:.1f} (uniq {uq:.1f})\tedges= {ne:.1f}\ttime= {t:.4f}\n{d:s}" \
          .format(b=FLAGS.b, a=i, c=totalscore/i, p=totalprec.prec(), 
                  ns=totalstates/i, uq=totaluniq/i, ne=totaledges/i, t=totaltime/i, 
                  d=totalprec.details())
    
    if FLAGS.uniqstat:
        for i in sorted(uniqstats):
            print >> logs, "%d\t%.1lf\t%d\t%d" % \
                  (i, sum(uniqstats[i]) / len(uniqstats[i]), \
                   min(uniqstats[i]), max(uniqstats[i]))

    if FLAGS.oracle:
        print >> logs, "oracle= ", totaloracle

if __name__ == "__main__":

    argv = FLAGS(sys.argv)
    if FLAGS.dp:
        FLAGS.newstate = True
    if FLAGS.newstate:
        from newstate import NewState as State
    else:
        from oldstate import State

    if FLAGS.profile:
        import cProfile as profile
        profile.run('main()', '/tmp/a')
        import pstats
        p = pstats.Stats('/tmp/a')
        p.sort_stats('time').print_stats(30)

    else:
        import time
        t_start = time.time()
        main()
        t_stop = time.time()
        print >>sys.stderr, t_stop-t_start

