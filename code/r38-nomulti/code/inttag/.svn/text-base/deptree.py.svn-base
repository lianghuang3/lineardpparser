#!/usr/bin/env python
from __future__ import division
import sys

import gflags as flags
FLAGS=flags.FLAGS

flags.DEFINE_boolean("shifttag", False, "shifttag")
flags.DEFINE_boolean("posttag", False, "posttag")
flags.DEFINE_boolean("pretag", False, "pretag")
flags.DEFINE_boolean("tag_end_symbol", True, "whether to tag </s>")
flags.DEFINE_integer("tag_qi", 0, "for shift-tag or pre-tag: tag q0, q1 or q2")

class DepVal(object):
    '''like PARSEVAL and BLEU, additive.'''

    __slots__ = "yes", "tot", "rootyes", "roottot", "completeyes", \
                "tagyes", "tagtot", "tagcompleteyes", "tagyes_unk", "tagtot_unk"

    def __init__(self, yes=0, tot=1e-10, rootyes=0, roottot=1e-11, completeyes=0, \
                 tagyes=0, tagtot=1e-10, tagcompleteyes=0, tagyes_unk=0, tagtot_unk=1e-10):
        self.yes = yes
        self.tot = tot
        self.rootyes = rootyes
        self.roottot = roottot
        self.completeyes = completeyes

        # lhuang: tagging
        self.tagyes = tagyes
        self.tagtot = tagtot
        self.tagcompleteyes = tagcompleteyes
        
        # yang: tagging for unks
        self.tagyes_unk = tagyes_unk
        self.tagtot_unk = tagtot_unk

    @staticmethod
    def unit(yes, tot, rootyes=False, tagyes=0, tagtot=1e-10, tagyes_unk=0, tagtot_unk=1e-10):
        ''' a single sentence '''
        return DepVal(yes, tot, int(rootyes), 1, int((yes == tot) and rootyes),
                      tagyes, tagtot, int(tagyes == tagtot),
                      tagyes_unk, tagtot_unk) # yang: for tag prec
        
    def prec(self):
        return self.yes / self.tot if self.tot != 0 else 0

    def prec100(self):
        return self.prec() * 100

    # yang: new function: for tag prec
    def tagprec(self):
        return self.tagyes / self.tagtot if self.tagtot != 0 else 0

    # yang: new function: for tag prec
    def tagprec100(self):
        return self.tagprec() * 100
        
    def tagprec_unk(self):
        return self.tagyes_unk / self.tagtot_unk if self.tagtot_unk != 0 else 0
        
    def tagprec100_unk(self):
        return self.tagprec_unk() * 100
        
    def tagprec_knw(self):
        tagtot_knw = self.tagtot - self.tagtot_unk
        return (self.tagyes - self.tagyes_unk) / tagtot_knw if tagtot_knw != 0 else 0
    
    def tagprec100_knw(self):
        return self.tagprec_knw() * 100

    def root(self):
        return self.rootyes / self.roottot if self.roottot != 0 else 0

    def complete(self):
        return self.completeyes / self.roottot if self.roottot != 0 else 0

    def nonroot(self):
        return (self.yes - self.rootyes) / (self.tot - self.roottot)

    def __iadd__(self, other):
        self.yes += other.yes
        self.tot += other.tot
        self.rootyes += other.rootyes
        self.roottot += other.roottot
        self.completeyes += other.completeyes

        self.tagyes += other.tagyes # yang: for tag prec
        self.tagtot += other.tagtot # yang: for tag prec
        self.tagcompleteyes += other.tagcompleteyes # yang: for tag prec
        self.tagyes_unk += other.tagyes_unk
        self.tagtot_unk += other.tagtot_unk
        
        return self

    def __add__(self, other):
        return DepVal(yes=self.yes+other.yes,
                      tot=self.tot+other.tot,
                      rootyes=self.rootyes+other.rootyes,
                      roottot=self.roottot+other.roottot,
                      completeyes=self.completeyes+other.completeyes,
                      tagyes=self.tagyes+other.tagyes, # yang: for tag prec
                      tagtot=self.tagtot+other.tagtot, # yang: for tag prec
                      tagcompleteyes=self.tagcompleteyes+other.tagcompleteyes,
                      tagyes_unk=self.tagyes_unk+other.tagyes_unk,
                      tagtot_unk=self.tagtot_unk+other.tagtot_unk) # yang: for tag prec

    def __eq__(self, other):
        return self.yes == other.yes and self.tot == other.tot

    def __cmp__(self, other):
        if isinstance(other, DepVal):
            return cmp(self.prec(), other.prec()) #TODO: use * not /
        else:
            return cmp(self.prec(), other)

    def __str__(self):
        return "{0:.2%}".format(self.prec())

    def details(self):
        return "word: %.2f%% (%d), non-root: %.2f%% (%d), root: %.2f%%, complete: %.2lf%% (%d), tagprec: %.2f%% (%d), tagcomplete: %d, tagprec_unk: %.2f%% (%d), tagprec_knw: %.2f%% (%d), " \
               % (self.prec100(), self.tot, \
                  self.nonroot()*100, self.tot - self.roottot,
                  self.root()*100, self.complete()*100, self.roottot,
                  self.tagprec100(), self.tagtot, self.tagcompleteyes,
                  self.tagprec100_unk(), self.tagtot_unk,
                  self.tagprec100_knw(), self.tagtot - self.tagtot_unk) # yang: for tag prec

class DepTree(object):

    __slots__ = "headidx", "lefts", "rights", "headtag"
    words = None # yang: for tag prec
    tags = None # yang: for pre-tag
    model = None

    def __eq__(self, other):
        return str(self) == str(other) ## TODO: CACHE

    def __init__(self, index, tag=None, lefts=[], rights=[]):
        self.headidx = index
        self.headtag = tag if tag is not None else DepTree.tags[index] # yang: for tag prec
        self.lefts = lefts
        self.rights = rights

    def head(self):
        return (self.words[self.headidx], self.headtag) # yang: for tag prec

    def tag(self):
        return self.head()[1]

    def word(self):
        return self.head()[0]

    # yang: for pretag plan
    def headi(self, i):
        return (self.words[self.headidx + i], self.tags[self.headidx + i]) \
               if self.headidx + i < len(self.words) else ('</s>', '</s>')

    # yang: for pretag plan               
    def wordi(self, i):
        return self.headi(i)[0]

    # yang: for pretag plan        
    def tagi(self, i):
        return self.headi(i)[1]

    def combine(self, next, action):
        ''' self and next are two consecutive elements on stack.
        self on the left, next on the right.'''
        if action[0] == 1: # left-reduce
            return DepTree(next.headidx, next.headtag, [self]+next.lefts, next.rights) # yang: for tag prec
        else:
            return DepTree(self.headidx, self.headtag, self.lefts, self.rights+[next]) # yang: for tag prec

    def __str__(self, wtag=True):
        ''' (... ... word/tag ... ...) '''

        # N.B.: string formatting is dangerous with single variable
        # "..." % var => "..." % tuple(var), because var might be list instead of tuple
        
        return "(%s)" % " ".join([x.__str__(wtag) for x in self.lefts] + \
                                 ["%s/%s" % tuple(self.head()) if wtag else self.word()] + \
                                 [x.__str__(wtag) for x in self.rights])

    def shortstr(self):
        return self.__str__(wtag=False)

    def wordtagpairs(self):
        ''' returns a list of word/tag pairs.'''
        return [x for y in self.lefts for x in y.wordtagpairs()] + \
               [self.head()] + \
               [x for y in self.rights for x in y.wordtagpairs()]

    @staticmethod
    def parse(line):
        ''' returns a tree and a sent. '''
        line += " " # N.B.
        words = [] # yang: for tag prec
        tags = [] # yang: for pre-tag
        _, t = DepTree._parse(line, 0, words, tags) # yang: for tag prec
        DepTree.words = words # yang: for tag prec
        DepTree.tags = tags
        return t #, sent

    @staticmethod
    def _parse(line, index, words, tags):
        ''' ((...) (...) w/t (...))'''

        assert line[index] == '(', "Invalid tree string %s at %d" % (line, index)
        index += 1
        head = None
        lefts = []
        rights = []
        while line[index] != ')':
            if line[index] == '(':
                index, t = DepTree._parse(line, index, words, tags)
                if head is None:
                    lefts.append(t)
                else:
                    rights.append(t)
                
            else:
                # head is here!
                rpos = min(line.find(' ', index), line.find(')', index))
                # see above N.B. (find could return -1)
                
                head = tuple(line[index:rpos].rsplit("/", 1))
                headidx = len(words)
                headtag = head[1] # yang: for tag prec
                words.append(head[0]) # yang: for tag prec
                tags.append(head[1]) # yang: for pre-tag
                index = rpos
                
            if line[index] == " ":
                index += 1

        assert line[index] == ')', "Invalid tree string %s at %d" % (line, index)
        t = DepTree(headidx, headtag, lefts, rights) # yang: for tag prec
        return index+1, t  ## N.B.: +1

    def is_punc(self):
        return self.tag() in [",", ".", ":", "``", "''", "-LRB-", "-RRB-", "PU"] # PU for CTB

    def links(self, is_root=True):
        '''returns a mapping of mod=>head'''

        m = {}
        iampunc = self.is_punc()
        for i, sub in enumerate(self.lefts + self.rights):
            if not sub.is_punc(): ## is that right?
                m[sub.headidx] = self.headidx
            subm = sub.links(is_root=False)
            for x, y in subm.iteritems():
                m[x] = y

        # root
        if is_root and not self.is_punc():
            m[self.headidx] = -1
        
        return m                

    def evaluate(self, other):
        '''returns precision, correct, all.'''

        if other is None:
            return DepVal()
        
        a = self.links()
        b = other.links()

        yes = 0.
        for mod, head in a.iteritems():
            #assert mod in b, "%s %s" % (str(self), str(other))
            if mod in b and head == b[mod]: # lhuang: a word can be mistagged as PU
                yes += 1

        # yang: tag prec
        tagyes, tagyes_unk, tagtot_unk = 0., 0., 0.
        for i, (x, y) in enumerate(zip(self.tagseq(), other.tagseq())):
            if self.words[i] not in self.model.knowns:
                tagtot_unk += 1        
            if x == y:
                tagyes += 1
                if self.words[i] not in self.model.knowns:
                    tagyes_unk += 1

        return DepVal.unit(yes, len(a), self.headidx==other.headidx, \
                           tagyes, len(self.tagseq()), tagyes_unk, tagtot_unk) # yang: for tag prec

    @staticmethod
    def compare(a, b):
        if a is not None:
            return a.evaluate(b)
        elif b is not None:
            return b.evaluate(a)
        else:
            return DepVal()

    def __len__(self):
        '''number of words'''
        return len(self.wordtagpairs())

    # yang: now action includes tag
    def _seq(self):
        ''' returns the oracle action (0,1,2) sequence (a list) for the deptree'''
        s = []

        for sub in self.lefts:
            s += sub._seq()

        if FLAGS.shifttag: # shift-tag
            # shift-tag has to tag </s> (FLAGS.tag_end_symbol cannot be applied)
            s += [(0, self.tagi(FLAGS.tag_qi), self.tag())] # shift myself
        elif FLAGS.posttag: # post-tag
            s += [(0, None)] # shift
            s += [(-1, self.tag())] # tag s0
        elif FLAGS.pretag: # pre-tag
            s += [(0, None)] # shift
            if FLAGS.tag_end_symbol or self.tagi(FLAGS.tag_qi + 1) != "</s>":
                s += [(-2, self.tagi(FLAGS.tag_qi + 1), self.tag())] # tag q0

        for _ in self.lefts:
            s += [(1, )] # left, in center-outward (reverse) order

        for sub in self.rights:
            s += sub._seq()
            s += [(2, )] # right, in center-outward (straight) order, so immediate reduce

        return s

    # yang
    def seq(self):
        if FLAGS.pretag:
            a = []
            for i in range(FLAGS.tag_qi):
                if i < len(DepTree.tags) - 1:
                    a += [(0, None), (-2, DepTree.tags[i + 1], "<s>")]
                else:
                    a += [(0, None), (-2, "</s>", "<s>")]
            return [(-2, DepTree.tags[0], "<s>")] + a + self._seq()
        elif FLAGS.shifttag:
            a = []
            for i in range(FLAGS.tag_qi):
                if i < len(DepTree.tags):
                    a += [(0, DepTree.tags[i], "<s>")]
                else:
                    a += [(0, "</s>", "<s>")]
            return a + self._seq()
        else:
            return self._seq()

    # yang: new function: for tag prec
    def replace_postags(self, tags):
        for left in self.lefts:
            left.replace_postags(tags)
        self.headtag = tags[self.headidx]
        for right in self.rights:
            right.replace_postags(tags)

    @staticmethod
    def load(filename):
        for i, line in enumerate(open(filename), 1):
            yield DepTree.parse(line)

    # yang: new function: for tag prec
    def tagseq(self):
        tags = []
        for left in self.lefts:
            tags += left.tagseq()
        tags.append(self.tag())
        for right in self.rights:
            tags += right.tagseq()
        return tags
        
    def wordseq(self):
        return " ".join(self.words)
            

if __name__ == "__main__":

    flags.DEFINE_boolean("seq", False, "print action sequence")
    flags.DEFINE_boolean("wordseq", False, "print word sequence")

    argv = FLAGS(sys.argv)
    
    for line in sys.stdin:
        tree = DepTree.parse(line)
        if FLAGS.seq:
            print tree.seq()
            print len(tree.seq())
        elif FLAGS.wordseq:
            print tree.wordseq()
        else:           
            print " ".join("%s/%s" % x for x in tree.wordtagpairs())
