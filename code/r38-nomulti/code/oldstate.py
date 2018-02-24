from svector import Vector
from model import Model
from deptree import DepTree
import gflags as flags
FLAGS=flags.FLAGS

class State(object):
    '''stack is a list of DepTrees.
       status=0/1/2=SH/LR/RR.
       score is forward (accumulative) cost.
       step is the number of steps.
       [i,j] span boundaries (next word is sent[j]).
       feats is features before concatenation with LEFT/RIGHT/SHIFT.
       leftptrs is the left pointers (states to my left that can combine w/ me): graph-structured stack.
       inside is the inside viterbi cost.
    '''

    names = ["SHIFT", "LEFT", "RIGHT"]
    mapnames = {"SHIFT":0, "LEFT":1, "RIGHT":2}
    
    __slots__ = "i", "j", "score", "action", "stack", "step", "_feats", "_featstr", \
                "leftptrs", "shiftcost", "inside", "_hash", "rank", "backptrs", "gold"

    sent = None ## cached for convenience (online use only)

    @staticmethod
    def setup():
        pass

    def __init__(self, i=0, j=0, action=0, stack=[], gold=False):
        self.step = 0
        self.j = j
        self.i = i
        self.score = 0
        self.action = action
        self.stack = stack
        self._feats = None
        self._featstr = None        
        self.inside = 0
        self.shiftcost = 0
        self._hash = None
        self.rank = -1
        self.leftptrs = None
        self.backptrs = None
        self.gold = gold

    @staticmethod
    def initstate():
        return State(gold=True)    

    def __cmp__(self, other):
        c = cmp(other.score, self.score)  # opt=MAX; N.B. must use cmp(a,b), can't use return a - b!!
        return c if c != 0 else cmp(other.inside, self.inside)

    def allowed_actions(self):
        ''' returns the set of allowed actions for the current state. '''
        a = []
        if self.j < len(self.sent):
            a += [0]
        if len(self.stack) >= 2: # not enough to REDUCE: only SHIFT
            a += [1,2]
        return a

    def take(self, action, action_gold=False):
        '''returns a list (iterator) of resulting states.'''

        if self.i == self.j == 0: ## don't count start
            actioncost = 0
        else:
            ## applying the model weights
            actioncost = self.feats(action).dot(self.model.weights) if self.model is not None else 0            
        
        if action == 0:  # SHIFT
            new = State(self.j, self.j+1, action, \
                        self.stack + [DepTree(self.j)])
            new.inside = 0
            self.shiftcost = actioncost # N.B.: self!
            new.score = self.score + actioncost # forward cost
            new.step = self.step + 1
            new.leftptrs = [self]
            new.backptrs = [(None, action)] # shift has no children

            new.gold = self.gold and action_gold   # gold is sequentially incremented

            yield new # shift always returns one unique offspring
            
        else:  # REDUCE
            for leftstate in self.leftptrs: # i'm combining with it
                newtree = leftstate.stack[-1].combine(self.stack[-1], action) # N.B.:theory! NOT self.stack[-2] with -1
                ## N.B.: theory! NOT self.stack[:-2] 
                new = State(leftstate.i, self.j, action, \
                            leftstate.stack[:-1] + [newtree])
                
                new.inside = leftstate.inside + self.inside + \
                             leftstate.shiftcost + actioncost # N.B.
        
                new.score = leftstate.score + self.inside + leftstate.shiftcost + actioncost #n.b.
                ## WRONG: new.score = self.score + actioncost # forward cost, only true for non-DP
                new.step = self.step + 1
                new.leftptrs = leftstate.leftptrs
                new.backptrs = [((leftstate, self), action)]

                # meaning of x.gold: first of all, viterbi-inside derivation is gold
                # and also, there is a gold path predicting x (same as earley item: bottom-up + top-down filtering)
                new.gold = leftstate.gold and self.gold and action_gold   # gold is binary
                
                yield new
        

    def top(self, index=0):
        return self.stack[-1-index] if len(self.stack) > index else None

    def qhead(self, index=0):
        return self.sent[self.j+index] if self.j + index < len(self.sent) else None

    def feats(self, action=None):
        if self._feats is None:
            self._feats = self.model.make_feats(self) #if self.model is not None else []

        if action is None:
            return self._feats
        else:
            aa = "=>" + State.names[action]
            fv = Vector()
            for f in self._feats:
                fv[f + aa] = 1

            return fv

    def featstr(self):
        if self._featstr is None:
            self._featstr = " ".join(self.feats(action=None))
        return self._featstr

    def __hash__(self):
        '''dynamic programming signature. see also __eq__().'''
        if self._hash is None:
            self._hash = hash(self.featstr())  # or tuple(...)
                
        return self._hash

    def __eq__(self, other):
        '''used in dynamic programming equivalence test. ordering still uses __cmp__.''' 
        return self.step == other.step and \
               self.i == other.i and self.j == other.j and \
               self._featstr == other._featstr

    def mergewith(self, other):
        '''merging equivalent states according to DP. THE CORE METHOD.'''

        # N.B.: we can prove that (VERY IMPORTANT):
        # if a state is shifted, then it merges with other's predictors (leftstates)
        # otherwise (reduced): no merging of leftptrs needed, but merge backptrs (insideptrs)
        
        if self.action == 0: # SHIFT:
            #assert self.leftptrs == other.leftptrs
            self.leftptrs.append(other.leftptrs[0]) # assert len(other.leftptrs)==1
            
        else: # REDUCE
            if FLAGS.forest or FLAGS.kbest >= 1:
                self.backptrs.append(other.backptrs[0])
        
#         if self.inside < other.inside:
#             print >> logs, "---", self
#             print >> logs, "+++", other
            
        
    def __str__(self, top=False):
        s = "*" if self.gold else " "
        s += "{0.step}({0.rank}): {0.score:6.2f} {0.inside:6.2f} : ({0.i}, {0.j})  ".format(self) + \
             " | ".join([(x.shortstr() if FLAGS.debuglevel>=2 else x.word()) for x in self.stack])
        if FLAGS.debuglevel >=1:
            s += "\t%s" % self.all_actions()
        if FLAGS.debuglevel >=2 and top:
            s += "\n" + "\n".join(["\t%5s on %s" % (State.names[ac], str(st)) \
                                   for (st, _, ac) in self.backptrs]) # BUGGY
        return s

    def __str__(self, top=False):
        s = "*" if self.gold else " "
        s += "{0.step}({0.rank}): {0.score:6.2f} {0.inside:6.2f} : ({0.i}, {0.j})  ".format(self)
        s += "(%d, %d) " % (len(self.leftptrs), len(self.backptrs))
        if FLAGS.debuglevel >=2:
            s += "\t%s" % self.all_actions()
        if FLAGS.debuglevel >=2 and top:
            s += "\t%s %s | %s" % (self.top(1).head()[0] if len(self.stack)>1 else "<s>",\
                                   self.top().head()[0] if len(self.stack)>0 else "<s>",\
                                   self.qhead()[0] if self.j < len(self.sent) else "</s>")
        return s
    
    def inside_actions(self):
## WRONG: only correct in non-dp
##         if self.backptrs is not None:
##             (st, ac) = self.backptrs[0]
##             return st.best_actions() + [State.names[ac]]

        if self.backptrs:            
            (children, action) = self.backptrs[0]
            this = [action] #[State.names[action]]            
            if children is None:
                return this # SHIFT
            else:
                left, right = children  # REDUCE
                return left.inside_actions() + right.inside_actions() + this
        else:
            return []

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

        return s

    def derivation_count(self, cache=None):
        ''' number of possible (inside) derivations '''

        if cache is None:
            cache = {}
        if self in cache:
            return cache[self]

        if self.action == 0:
            cache[self] = 1
        else:
            cache[self] = sum([left.derivation_count() * right.derivation_count() \
                               for ((left, right), action) in self.backptrs])
        return cache[self]
        
    def tree(self):
        return self.top() # top tree on the stack
