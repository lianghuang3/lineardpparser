import sys
logs = sys.stderr

import model
from collections import defaultdict

import gflags as flags
FLAGS=flags.FLAGS

from deptree import DepTree, DepVal

flags.DEFINE_boolean("use_gold_tags", False, "simulate training with gold_tags")
flags.DEFINE_boolean("limit_unk_tags", False, "limit the possible tags for unk words depending on its chars (only for Chinese)")
flags.DEFINE_boolean("presuf", False, "prefix and suffix of the words (for both Chinese and English)")

import copy

class NewState(object):
    '''stack is a list of DepTrees.
       status=0/1/2=SH/LR/RR.
       score is forward (accumulative) cost.
       step is the number of steps.
       [i,j] span boundaries (next word is sent[j]).
       feats is features before concatenation with LEFT/RIGHT/SHIFT.
       leftptrs is the left pointers (states to my left that can combine w/ me): graph-structured stack.
       inside is the inside viterbi cost.
    '''

    names = ["SHIFT", "LEFT", "RIGHT", "PRETAG", "POSTTAG"]
    mapnames = {"SHIFT":0, "LEFT":1, "RIGHT":2, "PRETAG":-2, "POSTTAG":-1} # yang: PRETAG added
    
    __slots__ = "i", "j", "score", "action", "step", "_feats", "_signature", \
                "leftptrs", "shiftcost", "inside", "_hash", "rank", "backptrs", "gold", \
                "s0", "s1", "s2", "s0lcrc", "s1lcrc", "nodeid", "headidx", \
                "qs", "leftmosttag" # lhuang: for shiftcost # yang: add q_1, q_2 and q_3

    words = None ## unk-sensitive mapped words cached for convenience (online use only) # yang
    featscache = defaultdict(list)
    actionfeatscache = defaultdict(lambda :model.new_vector())

    shared = 0
    tot = 0

    # cache gold tags for this sentence
    gold_tags = None

    # rel between Q and qs (-3 based) e.g. q0 = qs[3]
    # -----------------------------------------
    # FLAGS.tag_qi:      -1    0    1    2   .. 
    # -----------------------------------------
    # qs_idx:   0    1    2    3    4    5   .. (the index of qs)
    # -----------------------------------------
    #    Q  :  q_3  q_2  q_1  q0   q1   q2   .. (qs in Q)
    # -----------------------------------------
    qs_idx = 0
    qs_size = 0
    
    @staticmethod
    def setup():
##        NewState.featscache = defaultdict(list)
        NewState.actionfeatscache = {} #defaultdict(lambda :model.new_vector())
        NewState.qs_idx = FLAGS.tag_qi + 3
        NewState.qs_size = NewState.qs_idx + 2 + 1

    def __init__(self, step=0, i=0, j=0, action=(0, "<s>")):
        self._signature = None
        self._feats = None
        self._hash = None
        self.step = step
        self.j = j
        self.i = i
        self.rank = -1
        self.action = action
        self.shiftcost = defaultdict(float) # lhuang: shiftcost["JJ"]

    @staticmethod
    def initstate():
        s = NewState()
        s.j = s.i = -1 * (FLAGS.tag_qi) # yang: for tagging qi # tagging is tag_qi steps ahead of parsing
        s.gold = True
        s.score = 0
        s.inside = 0
        s.s0 = s.s1 = s.s2 = ("<s>", "<s>", "<s>", "<s>") # (word, word repre, tag, word cluster)
        s.s0lcrc = ("NONE", "NONE")
        s.s1lcrc = ("NONE", "NONE")
        s.leftptrs = None
        s.backptrs = None
        s.qs = [s.q(i) for i in range(-3, FLAGS.tag_qi + 2 + 1)] # from q_3 to q(FLAGS.tag_qi + 2)
        s.leftmosttag = "<s>" # lhuang: dp
        s.headidx = 0 # yang
        return s

    def __cmp__(self, other):
        c = cmp(other.score, self.score)  # opt=MAX; N.B. must use cmp(a,b), can't use return a - b!!
        return c if c != 0 else cmp(other.inside, self.inside)



    def allowed_actions(self):
        ''' returns the set of allowed actions for the current state. '''
        #                          Type of ACTIONs
        # ---------------------------------------------------------------------
        #  act type |         mode        |           representation
        # ---------------------------------------------------------------------
        #    TAG    |       PRE-TAG       | (-2, "</s>", s0's tag for recovery)
        #    TAG    |       POST-TAG      |              (-1, tag)             
        # SHIFT-TAG |      SHIFT-TAG      |    (0, tag, q0's tag for recovery) 
        #   SHIFT   | PRE-TAG & POST-TAG  |              (0, None)             
        #   REDUCE  |         ALL         |        (1, None) or (2, None)      
        a = []
        
        # [TAG ACTION]
        # SHIFT-TAG: no TAG action in shift-tag
        if FLAGS.shifttag:
            pass

        # PRE-TAG: TAG qi after SHIFT
        # if self.qs[NewState.qs_idx][2], the tag of the word being tagging, is None, must tag it immediately
        elif FLAGS.pretag and self.qs[NewState.qs_idx][2] is None:

            # FLAGS.tag_end_symbol: whether to tag the end symbol which may effect the final score
            if FLAGS.tag_end_symbol and self.qs[NewState.qs_idx][1] == "</s>":
                return [(-2, "</s>", self.s0[2])] # (TAG action, "</s>", s0's tag)
                                                  # s0's tag: for the recovery of the parse tree

            # FLAGS.limit_unk_tags: for unknown words in Chinese, only try the tags of its chars
            elif FLAGS.limit_unk_tags and self.qs[NewState.qs_idx][1] == "<unk>" and not FLAGS.use_gold_tags:
                tags = []
                for char in self.qs[NewState.qs_idx][0].decode("utf8"): # self.qs[NewState.qs_idx][0]: real word
                    char = char.encode("utf8")
                    if char in self.model.dict_char:
                        tags += self.model.dict_char[char]
                tags = list(set(tags)) # remove duplicates
                if len(tags) == 0: # if all the chars are unknown, try the tags of "word" <unk>
                    tags = self.model.dict_word[self.qs[NewState.qs_idx][1]]

            # try the tags using dict
            elif not FLAGS.use_gold_tags:
                tags = self.model.dict_word[self.qs[NewState.qs_idx][1]] # self.qs[NewState.qs_idx][1]: word repre of qi

            # use gold tags
            else:
                tags = [self.gold_tags[self.j + FLAGS.tag_qi]]

            for tag in tags:
                a += [(-2, tag, self.s0[2])] # (TAG action, tag, s0's tag)
            return a

##        # TODO post-tag: after shift, must tag s0            
##        if FLAGS.posttag and self.s0[2] is None:
##            tags = self.model.dict_word[self.s0[1]] if not FLAGS.use_gold_tags \
##                   else [self.gold_tags[self.j-1]]
##            for tag in tags:
##                a += [(-1, tag)] # tag s0
##            return a # only choice

        # [PARSE ACTION -- SHIFT]
        elif self.j < len(self.words):
            # SHIFT-TAG
            if FLAGS.shifttag:
                if self.qs[NewState.qs_idx][1] == "</s>": # the end symbol
                    a += [(0, "</s>", self.qs[3][2])] # (SHIFT-TAG action, "</s>", q0's tag), NOTE: qs[3] is q0
                else:
                    if FLAGS.use_gold_tags: # use gold tags
                        tags = [self.gold_tags[self.j + FLAGS.tag_qi]]
                    else:
                        tags = self.model.dict_word[self.qs[NewState.qs_idx][1]] # self.qs[NewState.qs_idx][1]: word repre of qi
                    for tag in tags:
                        a += [(0, tag, tag if FLAGS.tag_qi == 0 else self.qs[3][2])] # (SHIFT-TAG action, tag, q0's tag), NOTE: qs[3] is q0
        
            # PRE-TAG & POST-TAG 
            elif FLAGS.pretag or FLAGS.posttag:
                a += [(0, None)] # (SHIFT action, <stub>)

        # [PARSE ACTION -- REDUCE]
        if self.s1[2] != "<s>": # at least two items on stack?
            a += [(1, ), (2, )] # 1-LEFT  2-RIGHT
        return a

    def q(self, j=0):
        j += self.j
        if j < 0: # lhuang: q_1 (q_{-1})
            wt = ("<s>", "<s>", "<s>", "<s>")
        elif j >= len(self.words):
            wt = ("</s>", "</s>", None, "<s>") if FLAGS.tag_end_symbol \
                 else ("</s>", "</s>", "</s>", "</s>") # yang: also tag </s>
        else:
            #    (real word       , word repre      , tag , word cluster    )
            wt = (self.words[j][0], self.words[j][1], None, self.words[j][2])
        return wt

    # yang: used by shift or shift-tag
    def shift(self, action):
        new = NewState(self.step+1, self.j, self.j+1, action)
        new.headidx = self.j # TODO
        
        # for the stack
        # -----------
        # s2 s1 s0 q0
        # -----------
        new.s2 = self.s1
        new.s1 = self.s0
        new.s0lcrc = ("NONE", "NONE")
        new.s1lcrc = self.s0lcrc
        
        # for the queue
        # -----------------------
        # q_3 q_2 q_1 q0 q1 q2 ..
        # -----------------------
        new.qs = self.qs[1:] + self.q(FLAGS.tag_qi + 3)
        
##        [None for x in range(NewState.qs_size)]
##        for i in range(NewState.qs_size - 1):
##            new.qs[i] = self.qs[i + 1]
##        new.qs[NewState.qs_size - 1] = self.q(FLAGS.tag_qi + 3)

        new.s0 = self.qs[3] # NOTE: qs[3] is q0
        
        # SHIFT-TAG
        if FLAGS.shifttag:
            # lhuang: change tag
            new.qs[-1] = new.qs[-1][:2] + (action[1], new.qs[-1][3])
            
##            # if tag_qi is q0, the tags for both q_1 and s0 should be changed
##            if FLAGS.tag_qi == 0:
##                new.s0 = (new.s0[0], new.s0[1], action[1], new.s0[3])
                
##        # PRE-TAG
##        elif FLAGS.pretag:
##            pass
            
##        # POST-TAG
##        elif FLAGS.posttag:
##            new.s0[2] = action[1]

        return new

##    # TODO: lhuang: tag after shift: POST-TAG
##    def tag_s0(self, action):
##        new = NewState(self.step+1, self.i, self.j, action)
##        new.headidx = self.headidx
##        new.s0 = (self.s0[0], self.s0[1], action[1], self.s0[3])  # tag
##        new.s1 = self.s1
##        new.s2 = self.s2
##        new.q0 = self.q0
##        new.q1 = self.q1
##        new.q2 = self.q2
##        new.q3 = self.q3
##        new.q4 = self.q4
##        new.q_1 = new.s0 # yang
##        new.q_2 = self.q_2 # yang
##        new.q_3 = self.q_3 # yang
##        new.rank = -1 # lhuang: important!
##        new.s0lcrc = self.s0lcrc
##        new.s1lcrc = self.s1lcrc
##        new.action = action
##        return new        

    # used by PRE-TAG
    def tag_q(self, action):
        new = NewState(self.step+1, self.i, self.j, action)
        new.headidx = self.headidx
        
        # for the stack
        # --------
        # s2 s1 s0
        # --------
        new.s0 = self.s0
        new.s1 = self.s1
        new.s2 = self.s2
        new.s0lcrc = self.s0lcrc
        new.s1lcrc = self.s1lcrc
        
        # for the queue
        # -----------------------
        # q_3 q_2 q_1 q0 q1 q2 ..
        # -----------------------

        new.qs = self.qs[:]
        new.qs[NewState.qs_idx] = new.qs[NewState.qs_idx][:2] + \
                                  (action[1], new.qs[NewState.qs_idx][3])
        
##        new.qs = [None for x in range(NewState.qs_size)]
##        for i in range(NewState.qs_size):
##            new.qs[i] = self.qs[i]
        
##        # for the word being tagging
##        new.qs[NewState.qs_idx] = (new.qs[NewState.qs_idx][0], new.qs[NewState.qs_idx][1], action[1], new.qs[NewState.qs_idx][3])
        
        new.rank = -1 # lhuang: important!
        new.action = action
        return new

    def reduce(self, left, action):
        new = NewState(self.step+1, left.i, self.j, action)
            
        # for the stack
        # --------
        # s2 s1 s0
        # --------
        new.s1 = left.s1 # not self.s2!
        new.s2 = left.s2
        new.s0 = self.s0 if action[0] == 1 else left.s0 # not self.s1!
        new.headidx = self.headidx if action[0] == 1 else left.headidx # TODO
        new.s0lcrc = (left.s0[2], self.s0lcrc[1]) if action[0] == 1 else \
                     (left.s0lcrc[0], self.s0[2])
        new.s1lcrc = left.s1lcrc
        
        # for the queue
        # -----------------------
        # q_3 q_2 q_1 q0 q1 q2 ..
        # -----------------------
        news.qs = self.qs[:]
        
##        new.qs = [None for x in range(NewState.qs_size)]
##        for i in range(NewState.qs_size):
##            new.qs[i] = self.qs[i]        
        
        return new
        
    def take(self, action, action_gold=False):
        '''returns a list (iterator) of resulting states.'''

        # lhuang: always evaluate the first step (used to be zero cost)
        if False and self.i == self.j == 0: ## don't count start
            actioncost = 0
        else:
##            actioncost = self.feats(action).dot(self.model.weights) if self.model is not None else 0            
            actioncost = self.featscore(action) if self.model is not None else 0            
            
        # SHIFT action
        if action[0] == 0:
            tag = action[1] # for SHIFT-TAG, this must not be None
            new = self.shift(action)
            
            new.inside = 0
            self.shiftcost[tag] = actioncost # N.B.: self! and shiftcost["JJ"]

            new.score = self.score + actioncost # forward cost
            new.leftptrs = [self]
            new.backptrs = [(None, action, 0)] # shift has no children
                
            new.gold = self.gold and action_gold   # gold is sequentially incremented
            new.leftmosttag = tag
                
            yield new
        
        # TAG action
        elif action[0] == -1 or action[0] == -2:
            tag = action[1]
            new = self.tag_s0(action) if action[0] == -1 else self.tag_q(action)
            new.inside = 0
            new.shiftcost[tag] = actioncost
            new.score = self.score + actioncost
            new.leftptrs = self.leftptrs # lhuang: skip shift
            new.backptrs = [(None, action, 0)] # shift has no children
                
            new.gold = self.gold and action_gold   # gold is sequentially incremented
            new.leftmosttag = tag
                
            yield new            
            
        # REDUCE action
        else:
            for leftstate in self.leftptrs: # i'm combining with it
                new = self.reduce(leftstate, action)

                # lhuang: shiftcost depends on shift tag
                shiftcost = leftstate.shiftcost[self.leftmosttag]
                new.inside = leftstate.inside + self.inside + \
                             shiftcost + actioncost # N.B.
                if FLAGS.dp:
                    new.score = leftstate.score + self.inside + shiftcost + actioncost #n.b.
                else:
                    new.score = self.score + actioncost
                    
                new.leftptrs = leftstate.leftptrs
                new.backptrs = [((leftstate, self), action, shiftcost + actioncost)]

                new.gold = leftstate.gold and self.gold and action_gold   # gold is binary
                new.leftmosttag = leftstate.leftmosttag

                yield new

    def signature(self):
        # TODO: for joint tagging: remove the [0]
        if self._signature is None:
            # TODO: self.qs[:NewState.qs_idx + 1] for both PRE-TAG and SHIFT-TAG? ok
            self._signature = (self.j, self.s2[2]) + self.s1 + self.s0 + \
                               tuple(self.qs[:NewState.qs_idx + 1]) + \
                               self.s0lcrc + self.s1lcrc + (self.leftmosttag,) # q1/q2 not needed # lhuang: leftmosttag important
        return self._signature

    def __hash__(self):
        '''dynamic programming signature. see also __eq__().'''
        if self._hash is None:
            self._hash = hash(self.signature())  # or tuple(...)
                
        return self._hash

    def __eq__(self, other):
        '''used in dynamic programming equivalence test. ordering still uses __cmp__.''' 
        return self._signature == other._signature and \
               self.i == other.i

    def mergewith(self, other):
        '''merging equivalent states according to DP. THE CORE METHOD.'''

        if self.action[0] == 0: # SHIFT:
            self.leftptrs.append(other.leftptrs[0]) # assert len(other.leftptrs)==1
            
        else: # REDUCE
            if FLAGS.forest or FLAGS.oracle or FLAGS.kbest >= 1:
                self.backptrs.append(other.backptrs[0])
        
    def __str__(self, top=False):
        s = "*" if self.gold else " "
        s += "{0.step}({0.rank}): {0.score:6.2f} {0.inside:6.2f} : ({0.i}, {0.j}) ".format(self)
        s += "(l%d, b%d) " % (len(self.leftptrs), len(self.backptrs)) \
            if self.leftptrs is not None and self.backptrs is not None else "" # yang: check if it is None
        if FLAGS.debuglevel >=2:
            s += "\t%s" % self.all_actions()
        if FLAGS.debuglevel >=2 and top:
            # TODO: debug for qs
            s += "\t %s/%s %s/%s | %s/%s" % (self.s1[0], self.s1[2], \
                                             self.s0[0], self.s0[2], \
                                             self.q0[0], self.q0[2]) #, self.s0lcrc, self.s1lcrc)
        return s

    def inside_actions(self):

        if self.backptrs:            
            (children, action, _) = self.backptrs[0]
            this = [action] #[State.names[action]]            
            if children is None:
                # lhuang
                if FLAGS.posttag or FLAGS.pretag: # yang
                    if action[0] == 0:
                        return this  
                    else:
                        return [(0, None)] + this  # shift, tag
                else:
                    return this # SHIFT
            else:
                left, right = children  # REDUCE
                return left.inside_actions() + right.inside_actions() + this
        else:
            return []

    def tree(self):
        '''alternatively: simulate actions'''

        (children, action, _) = self.backptrs[0]
        if children is None:
            if FLAGS.pretag and action[0] == -2:
                return DepTree(self.i, action[2])
            elif FLAGS.shifttag and action[0] == 0:
                return DepTree(self.i, action[2])
            else:
                return DepTree(self.i, action[1]) # yang: SHIFT tag (changes tag in deptree)
        else:
            left, right = children  # REDUCE            
            return DepTree.combine(left.tree(), right.tree(), action)

    def all_actions(self):
        s = []
        item = self
        while item:
#             if FLAGS.debuglevel >=1:
#                 print >> logs, item, s
            s = item.inside_actions() + s
            if item.leftptrs:
                # N.B.: only approximately true.
                # for incomplete derivations, this is just a guess of 1-best (as it's most likely)                
                item = item.leftptrs[0] 
            else:
                break

        return s[1:] if FLAGS.pretag else s # yang: no SHIFT before the first tag_q0

    def derivation_count(self, cache=None):
        ''' number of possible (inside) derivations '''

        if cache is None:
            cache = {}
        if self in cache:
            return cache[self]

        if self.action[0] == 0:
            cache[self] = 1
        else:
            cache[self] = sum(left.derivation_count() * right.derivation_count() \
                              for ((left, right), action, _) in self.backptrs)
        return cache[self]

    def previsit(self, cache=None, cache2=None):
        ''' just count how many nodes are reachable (useful for forest) '''

        if cache is None:
            cache = set()
        if cache2 is None:
            cache2 = set()

        sig = (self.step, self.rank)
        if sig in cache:
            return 

        cache.add(sig)

        if self.action[0] != 0:
            for ((left, right), action, _) in self.backptrs:
                left.previsit(cache, cache2)
                right.previsit(cache, cache2)

        cache2.add(sig)
        self.nodeid = len(cache2)

    def postvisit(self, cache=None):
        ''' real dumping '''

        if cache is None:
            cache = set()

        sig = (self.step, self.rank)
        if sig in cache:
            return 

        cache.add(sig)

        if self.action[0] != 0:
            for ((left, right), action, _) in self.backptrs:
                left.postvisit(cache)
                right.postvisit(cache)

            c = len(self.backptrs)
        else:
            c = 0 # SHIFT STATE
            
        print "%d\t%d [%d-%d]\t%d ||| %s/%s=1" % (self.nodeid, self.headidx, self.i, self.j, c,\
                                                  self.s0[0], self.s0[2])
        if self.action[0] != 0:
            for ((left, right), action, addcost) in self.backptrs:
                print "\t%d %d ||| 0=%f dir=%s" % (left.nodeid, right.nodeid, \
                                                    addcost, \
                                                    action)

    def besteval(self, reflinks, cache=None):
        ''' sub-oracle '''

        if cache is None:
            cache = {}

        sig = (self.step, self.rank)
        if sig in cache:
            return cache[sig]        
            
        if self.action[0] == 0:
            s = DepVal()
            t = self.tree()
        else:
            h = self.headidx
            s = -1
            t = None
            for ((left, right), action, _) in self.backptrs:
                # yang: action is a tuple
                m = left.headidx if action[0] == 1 else right.headidx
                this = 1 if (m in reflinks and reflinks[m] == h) else 0
                thistot = 1 if (m in reflinks) else 0
                
                lefteval, lefttree = left.besteval(reflinks, cache)
                righteval, righttree = right.besteval(reflinks, cache)

                thiseval = DepVal(yes=this, tot=thistot) + lefteval + righteval
                
                if thiseval > s:
                    s = thiseval
                    t = DepTree.combine(lefttree, righttree, action)
                    
        cache[sig] = s, t
        return s, t
        
    def make_feats(self, action):
        # lhuang: step=0 is non-deterministic now!
#        if self.step == 0:
#            return []
        #print self.q0, self.q1, self.q2, self.q3, self.q4, self.q_1, self.q_2, self.q_3, self.s0, self.s1, self.s2, self.s0lcrc, self.s1lcrc 
        if self._feats is None:
            # use automatically generated code
            # q(-1) -2 -3 only uses w, not t, so it's ok
            # yang: use separate templates for tagging
            
            # TAG action
            if action[0] == -1 or action[0] == -2:
                # feature window
                # ----------------
                # b_2 b_1 b0 b1 b2  NOTE: b0 is q[FLAGS.tag_qi]
                # ----------------
                self._feats = self.model.eval_module.\
                              static_eval_tag(self.s0, self.s1, self.s2,
                                              self.s0lcrc, self.s1lcrc,
                                              self.qs[NewState.qs_idx - 2], self.qs[NewState.qs_idx - 1],
                                              self.qs[NewState.qs_idx],
                                              self.qs[NewState.qs_idx + 1], self.qs[NewState.qs_idx + 2])
                                              
                # word morphology features for tagging (for both Chinese and English)
                # if FLAGS.presuf is on, use morphology feats for all the words                                        
                if FLAGS.presuf and self.qs[NewState.qs_idx][1] != "</s>" and self.qs[NewState.qs_idx][1] != "<s>":
                    char_feats = []
                    chars = self.qs[NewState.qs_idx][0].decode("utf8")
                    for i in range(min(4, len(chars))):
                        char_feats += ["pre=%s" % (chars[:i+1].encode("utf8"))]
                        char_feats += ["suf=%s" % (chars[-1-i:].encode("utf8"))]
                    self._feats += char_feats
                    
            # SHIFT & REDUCE action        
            else:
                self._feats = self.model.eval_module.\
                                  static_eval(self.s0, self.s1, self.s2,
                                              self.s0lcrc, self.s1lcrc,
                                              self.qs[3], self.qs[4], self.qs[5]) # q0, q1, q2. NOTE: qs[3] is q0

                # TODO: MERGE W/ ABOVE
                
                # word morphology features for parsing (for both Chinese and English)
                # if FLAGS.presuf is on, use morphology feats for all the words                                        
                if FLAGS.presuf and self.qs[NewState.qs_idx][1] != "</s>" and self.qs[NewState.qs_idx][1] != "<s>":
                    char_feats = []
                    chars = self.qs[NewState.qs_idx][0].decode("utf8")
                    for i in range(min(4, len(chars))):
                        char_feats += ["pre=%s" % (chars[:i+1].encode("utf8"))]
                        char_feats += ["suf=%s" % (chars[-1-i:].encode("utf8"))]
                    self._feats += char_feats

        # yang: special treatment for SHIFT
        aa = "=>" + NewState.names[action[0]]
        if (not FLAGS.use_gold_tags) and \
               ((FLAGS.shifttag and action[0] == 0) or \
                (FLAGS.posttag and action[0] == -1) or \
                (FLAGS.pretag and action[0] == -2)): # shift-tag, post-tag, pre-tag
            aa += ("_" + action[1]) # PLUS TAG
            
        if self.model.unkdel:
            return [(ff + aa) for ff in self._feats if ff.find("<unk>") == -1] # no unks!
        else:
            return [(ff + aa) for ff in self._feats]

    def featscore(self, action):
        # lhuang: no special treatment for first step!
#        if self.step == 0:
#            return 0 # model.new_vector()
        sig = (action, self.signature())
        NewState.tot += 1
        if sig in NewState.actionfeatscache:
            NewState.shared += 1  
            return NewState.actionfeatscache[sig]

        if FLAGS.debuglevel >= 3:
            print >> logs, zip(self.make_feats(action), \
                               ["%.2f" % self.model.weights[ff] for ff in self.make_feats(action)])
        
        score = sum([self.model.weights[ff] for ff in self.make_feats(action)])

        NewState.actionfeatscache[sig] = score
        return score
