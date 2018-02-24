#!/usr/bin/env python

from __future__ import division

import sys
import math
logs = sys.stderr
from collections import defaultdict

import time
from mytime import Mytime

import gflags as flags
FLAGS=flags.FLAGS

flags.DEFINE_string("weights", None, "weights file (feature instances and weights)", short_name="w")
flags.DEFINE_boolean("svector", False, "use David's svector (Cython) instead of Pythonic defaultdict")
flags.DEFINE_boolean("featstat", False, "print feature stats")
flags.DEFINE_string("outputweights", None, "write weights (in short-hand format); - for STDOUT", short_name="ow")
flags.DEFINE_boolean("autoeval", True, "use automatically generated eval module")
flags.DEFINE_integer("unk", 0, "treat words with count less than COUNT as UNKNOWN")
flags.DEFINE_boolean("debug_wordfreq", False, "print word freq info")
flags.DEFINE_boolean("unktag", False, "use POS tags for unknown words")
flags.DEFINE_boolean("unkdel", False, "remove features involving unks")
flags.DEFINE_integer("tagunk", 1, "treat words with count less than COUNT as UNKNOWN") # yang: add <unk>
flags.DEFINE_integer("tagunk_test", 1, "treat words with count less than COUNT as UNKNOWN") # yang: add <unk>

flags.DEFINE_string("dict", None, "dictionary file") # lhuang: moved here from parser.py
flags.DEFINE_string("wc", None, "word cluster file") # yang: word cluster

def new_vector():
    return defaultdict(int) if not FLAGS.svector else svector.Vector() # do not use lambda 

class Model(object):
    '''templates and weights.'''

##    __slots__ = "templates", "weights", "list_templates", "freq_templates"

    names = ["SHIFT", "LEFT", "RIGHT"]
    indent = " " * 4
    eval_module = None # by default, use my handwritten static_eval()
    
    def __init__(self, weightstr):

        self.knowns = set()
        self.wc = defaultdict(lambda: "<null>")
        self.unk = FLAGS.unk
        self.unktag = FLAGS.unktag
        self.unkdel = FLAGS.unkdel
        assert not (self.unkdel and self.unktag), "UNKDEL and UNKTAG can't be both true"

        if FLAGS.svector: # now it is known
            global svector
            try:
                svector = __import__("svector")
                print >> logs, "WARNING: using David's svector (Cython). Performance might suffer."
            except:
                print >> logs, "WARNING: failed to import svector. using Pythonic defaultdict instead (actually faster)."
                FLAGS.svector = False # important

        self.templates = {} # mapping from "s0t-q0t" to the eval expression
        self.list_templates = [] # ordered list of template keys "s0t-q0t"
        self.freq_templates = defaultdict(int)
        self.weights = new_vector() #Vector()

        # yang: separate templates for tagging
        self.templates_tag = {}
        self.list_templates_tag = []
        self.freq_templates_tag = defaultdict(int)

        self.read_weights(weightstr)
##        self.featurenames = set(self.weights.iterkeys())

        if FLAGS.featstat:
            self.print_templates()

        # yang
        if FLAGS.dict is not None:
            self.read_dict()

        if FLAGS.wc is not None:
            self.read_wc()

    # yang: word cluster
    def read_wc(self):
        for line in open(FLAGS.wc):
            line = line.strip()
            if line == '':
               continue
            wc, word, _ = line.split("\t")
            self.wc[word] = wc

    # yang: new function: dict
    def read_dict(self):
##        default_unktags = ['FW', 'JJ', 'NN', 'NR', 'NT', 'VV']
        default_unktags = ['NN', 'CD', 'VV', 'NR', 'JJ', 'AD', 'VA', 'NT']
        dict_word = defaultdict(lambda: default_unktags) # lhuang: yang's original unk tags
        self.dict_word = {}
        self.dict_char = {}
        flag = 0
        for line in open(FLAGS.dict):
            line = line.strip()
            if line == '':
                continue
            if line == '------':
                flag = 1
                continue
                
            if flag == 0:
                # <unk>  745    NN 234  VV 124  NR 123 ...
                # word   freq   t1 f1   t2 f2   t3 f3  ...
                word, tag_freqs = line.split("\t", 1)
                self.dict_word[word] = [x.split()[0] for x in tag_freqs.split("\t")[1:]] # omit freq
                if word != "<unk>":
                    self.knowns.add(word)
                else:
                    # lhuang: in case <unk>  0   
                    for t in default_unktags:
                        if t not in self.dict_word[word]:
                            self.dict_word[word].append(t)
            elif flag == 1:
                # char   freq   t1 f1   t2 f2   t3 f3  ...
                char, tag_freqs = line.split("\t", 1)
                self.dict_char[char] = [x.split()[0] for x in tag_freqs.split("\t")[1:]] # omit freq

        print >> logs, self.dict_word["<unk>"]

    def map_unk_wc(self, word):
        w = word if word in self.knowns else "<unk>"
        wc = self.wc[word]
        return (word, w, wc)

    def count_knowns_from_train(self, trainfile, devfile):
        '''used in training'''

        print >> logs, "counting word freqs from %s, unktag=%s" % (trainfile, self.unktag)
        stime = time.time()

        words = defaultdict(int)        
        for i, line in enumerate(open(trainfile)):
            for word in line.split():
                word = word.strip("()").rsplit("/", 1)[0]
                words[word] += 1

        if FLAGS.debug_wordfreq:
            devunk1 = set()
            devunk0 = set()
            for line in open(devfile):                
                for word in line.split():
                    word = word.strip("()").rsplit("/", 1)[0]
                    if words[word] <= self.unk and words[word] > 0:
                        devunk1.add(word)
                    if words[word] == 0:
                        devunk0.add(word)
                        
            print >> logs, "=1", len(devunk1), " ".join(sorted(devunk1))
            print >> logs
            print >> logs, "=0", len(devunk0), " ".join(sorted(devunk0))

##            freqs = defaultdict(list)
##            for word, freq in words.items():
##                freqs[freq].append(word)

##            for freq in sorted(freqs, reverse=True):
##                print >> logs, freq, len(freqs[freq]), " ".join(sorted(freqs[freq]))
##                print >> logs

        self.knowns = set()
        for word, freq in words.items():
            if freq > self.unk:
                self.knowns.add(word)

        print >> logs, "%d lines: %d known (freq > %d), %d unknown. counted in %.2f seconds" % \
              (i+1, len(self.knowns), self.unk, len(words)-len(self.knowns), time.time() - stime)
##        print >> logs, " ".join(sorted(self.knowns))

    def add_template(self, template_type, s, freq=1): # yang: add template_type
        ## like this: "s0w-s0t=%s|%s" % (s0w, s0t) 
        symbols = s.split("-") # static part: s0w-s0t

        # yang: add templates for tagging and parsing respectively
        (templates, list_templates, freq_templates) = \
            (self.templates, self.list_templates, self.freq_templates) \
            if template_type == 0 else \
            (self.templates_tag, self.list_templates_tag, self.freq_templates_tag)

        if s not in templates:
            tmp = '"%s=%s" %% (%s)' % (s, \
                                       "|".join(["%s"] * len(symbols)), \
                                       ", ".join(symbols))
            
            templates[s] = compile(tmp, "2", "eval")
            
            list_templates.append((s, tmp)) # in order

        freq_templates[s] += int(freq)

    def print_autoevals(self):

        tfilename = str(int(time.time()))
        templatefile = open("/tmp/%s.py" % tfilename, "wt")
        
        print >> templatefile, "#generated by model.py"
        print >> templatefile, "import sys; print >> sys.stderr, 'importing succeeded!'"
        print >> templatefile, "def static_eval((s0W, s0w, s0t, s0wc), \
                                                (s1W, s1w, s1t, s1wc), \
                                                (s2W, s2w, s2t, s2wc), \
                                                (s0lct, s0rct), (s1lct, s1rct), \
                                                (q0W, q0w, q0t, q0wc), \
                                                (q1W, q1w, q1t, q1wc), \
                                                (q2W, q2w, q2t, q2wc)):"
        
        print >> templatefile,  "%sreturn [" % Model.indent
        
        for s, e in self.list_templates:
            print >> templatefile, "%s%s," % (Model.indent * 2, e)
        
        print >> templatefile, "%s]" % (Model.indent * 2)

        # yang: add static_eval_tag() for tag
        print >> templatefile, "def static_eval_tag((s0W, s0w, s0t, s0wc), \
                                                 (s1W, s1w, s1t, s1wc), \
                                                 (s2W, s2w, s2t, s2wc), \
                                                 (s0lct, s0rct), (s1lct, s1rct), \
                                                 (b_2W, b_2w, b_2t, b_2wc), \
                                                 (b_1W, b_1w, b_1t, b_1wc), \
                                                 (b0W, b0w, b0t, b0wc), \
                                                 (b1W, b1w, b1t, b1wc), \
                                                 (b2W, b2w, b2t, b2wc)):"
        print >> templatefile,  "%sreturn [" % Model.indent
        for s, e in self.list_templates_tag:
            print >> templatefile, "%s%s," % (Model.indent * 2, e)
        
        print >> templatefile, "%s]" % (Model.indent * 2)


        templatefile.close()

        if FLAGS.autoeval:
            sys.path.append('/tmp/')
            print >> logs, "importing auto-generated file /tmp/%s.py" % tfilename
            # to be used in newstate
            Model.eval_module = __import__(tfilename)
        else:
            Model.eval_module = Model        
        
    def print_templates(self, f=logs):
        print >> f, ">>> %d templates in total:" % len(self.templates)
        print >> f, "\n".join(["%-20s\t%d" % (x, self.freq_templates[x]) \
                               for x, _ in self.list_templates])

        # yang: print templates for tagging
        print >> f, "<<< %d templates for tagging:" % len(self.templates_tag)
        print >> f, "\n".join(["%-20s\t%d" % (x, self.freq_templates_tag[x]) \
                               for x, _ in self.list_templates_tag])

        print >> f, "---"

    def read_templates(self, filename):

        ## try interpreting it as a filename, if failed, then as a string
        try:
            f = open(filename)
            print >> logs, "reading templates from %s" % filename,
            for x in f:
                x = x.strip()
                if x == "":
                    continue
                if x[0] == "#":
                    continue
                if x[:3] == "---":
                    break
                if x[:3] == ">>>": # yang: templates for parsing
                    template_type = 0
                    continue
                if x[:3] == "<<<": # yang: templates for tagging
                    template_type = 1
                    continue
                try:
                    s, freq = x.split()
                except:
                    s, freq = x, 1
                self.add_template(template_type, s, freq) # yang: add template_type
 
        except:
            ## from argv string rather than file
            for x in filename.split():
                self.add_template(x)
            f = None

        print >> logs, "%d feature templates read (%d for parsing, %d for tagging)." \
                       % (len(self.templates) + len(self.templates_tag), len(self.templates), len(self.templates_tag))
        return f

    def read_weights(self, filename, infertemplates=False):
        '''instances are like "s0t-q0t=LRB-</s>=>LEFT     3.8234"'''

        infile = self.read_templates(filename)

        infertemplates = len(self.templates) < 1
        if infertemplates:
            print >> logs, "will infer templates from weights..."        

        mytime = Mytime()
        i = 0
        if infile is not None:
            print >> logs, "reading feature weights from %s\t" % filename,
            for i, line in enumerate(infile, 1):
                if i % 200000 == 0:
                    print >> logs, "%d lines read..." % i,

                if line[0] == " ":
                    # TODO: separate known words line (last line)
                    self.knowns = set(line.split())
                    print >> logs, "\n%d known words read." % len(self.knowns)
                    self.unk = 1 # in cae you forgot to say it; doesn't matter 1 or x
                    break
                
                feat, weight = line.split() 
                self.weights[feat] = float(weight)

                if infertemplates:
                    self.add_template(feat.split("=", 1)[0], 1) ## one occurrence

        print >> logs, "\n%d feature instances (%d lines) read in %.2lf seconds." % \
              (len(self.weights), i, mytime.period())


        self.print_autoevals()

    def make_feats(self, state):
        '''returns a *list* of feature templates for state.'''
        
        fv = new_vector() #Vector()
        top = state.top()
        topnext = state.top(1)
        top3rd = state.top(2)
        qhead = state.qhead()
        qnext = state.qhead(1)

        ## this part is manual; their combinations are automatic
        s0 = top.head() if top is not None else ("<s>", "<s>") # N.B. (...)
        s1 = topnext.head() if topnext is not None else ("<s>", "<s>") 
        s2 = top3rd.head() if top3rd is not None else ("<s>", "<s>") 

        q0 = qhead if qhead is not None else ("</s>", "</s>") 
        q1 = qnext if qnext is not None else ("</s>", "</s>")

        s0lct = top.lefts[0].tag() if (top is not None and len(top.lefts) > 0) else "NONE"
        s0rct = top.rights[-1].tag() if (top is not None and len(top.rights) > 0) else "NONE"
        s1lct = topnext.lefts[0].tag() if (topnext is not None and len(topnext.lefts) > 0) else "NONE"
        s1rct = topnext.rights[-1].tag() if (topnext is not None and len(topnext.rights) > 0) else "NONE"
        
        ## like this: "s0w-s0t=%s|%s" % (s0w, s0t) ---> returns a list here!
        return Model.static_eval(q0, q1, s0, s1, s2, (s0lct, s0rct), (s1lct, s1rct))
#        return [eval(t) for t in self.templates.values()] ## eval exprs are the values, not keys

    def write(self, filename="-", weights=None):

        if weights is None:
            weights = self.weights

        if filename == "-":
            outfile = sys.stdout
            filename = "STDOUT"  # careful overriding
        else:
            outfile = open(filename, "wt")

        self.print_templates(outfile)

        mytime = Mytime()

        nonzero = 0
        print >> logs, "sorting %d features..." % len(weights),
        for i, f in enumerate(sorted(weights), 1):
            if i == 1: # sorting done
                print >> logs, "done in %.2lf seconds." % mytime.period()
                print >> logs, "writing features to %s..." % filename
                
            v = weights[f]
            if math.fabs(v) > 1e-3:
                print >> outfile, "%s\t%.5lf" % (f, v)
                nonzero += 1

        if self.unk > 0: # print known words
            print >> outfile, " " + " ".join(sorted(self.knowns)) # " " to mark

        print >> logs, "%d nonzero feature instances written in %.2lf seconds." % \
              (nonzero, mytime.period())  ## nonzero != i

    @staticmethod
    def trim(fv):
        for f in fv:
            if math.fabs(fv[f]) < 1e-3:
                del fv[f]
        return fv

    @staticmethod
    def truncate_refseq(refseq, myseq):
        num_tag, num_parse = 0, 0
        for action in myseq:
            if action[0] == -2 or action[0] == -1:
                num_tag += 1
            elif action[0] == 0 or action[0] == 1 or action[0] == 2:
                num_parse += 1

        trunc_refseq = []
        for action in refseq:
            if num_tag > 0 or num_parse > 0:
                trunc_refseq.append(action)
                if action[0] == -2 or action[0] == -1:
                    num_tag -= 1
                elif action[0] == 0 or action[0] == 1 or action[0] == 2:
                    num_parse -= 1
        return trunc_refseq

    @staticmethod
    def static_eval((q0w, q0t), (q1w, q1t), (s0w, s0t), (s1w, s1t), (s2w, s2t), (s0lct, s0rct), (s1lct, s1rct)):
        return ["q0t=%s" % (q0t),
                "q0w-q0t=%s|%s" % (q0w, q0t),
                "q0w=%s" % (q0w),
                "s0t-q0t-q1t=%s|%s|%s" % (s0t, q0t, q1t),
                "s0t-q0t=%s|%s" % (s0t, q0t),
                "s0t-s1t=%s|%s" % (s0t, s1t),
                "s0t-s1w-s1t=%s|%s|%s" % (s0t, s1w, s1t),
                "s0t=%s" % (s0t),
                "s0w-q0t-q1t=%s|%s|%s" % (s0w, q0t, q1t),
                "s0w-s0t-s1t=%s|%s|%s" % (s0w, s0t, s1t),
                "s0w-s0t-s1w-s1t=%s|%s|%s|%s" % (s0w, s0t, s1w, s1t),
                "s0w-s0t-s1w=%s|%s|%s" % (s0w, s0t, s1w),
                "s0w-s0t=%s|%s" % (s0w, s0t),
                "s0w-s1w-s1t=%s|%s|%s" % (s0w, s1w, s1t),
                "s0w-s1w=%s|%s" % (s0w, s1w),
                "s0w=%s" % (s0w),
                "s1t-s0t-q0t=%s|%s|%s" % (s1t, s0t, q0t),
                "s1t-s0t-s0lct=%s|%s|%s" % (s1t, s0t, s0lct),
                "s1t-s0t-s0rct=%s|%s|%s" % (s1t, s0t, s0rct),
                "s1t-s0w-q0t=%s|%s|%s" % (s1t, s0w, q0t),
                "s1t-s0w-s0lct=%s|%s|%s" % (s1t, s0w, s0lct),
                "s1t-s1lct-s0t=%s|%s|%s" % (s1t, s1lct, s0t),
                "s1t-s1lct-s0w=%s|%s|%s" % (s1t, s1lct, s0w),
                "s1t-s1rct-s0t=%s|%s|%s" % (s1t, s1rct, s0t),
                "s1t-s1rct-s0w=%s|%s|%s" % (s1t, s1rct, s0w),
                "s1t=%s" % (s1t),
                "s1w-s1t=%s|%s" % (s1w, s1t),
                "s1w=%s" % (s1w),
                "s2t-s1t-s0t=%s|%s|%s" % (s2t, s1t, s0t)]

    def prune(self, filenames):
        '''prune features from word/tag lines'''

        print >> logs, "pruning features using %s..." % filenames,
        
        fullset = set()
        for filename in filenames.split():
            for l in open(filename):
                for w, t in map(lambda x:x.rsplit("/", 1), l.split()):
                    fullset.add(w)
                    fullset.add(t)

        print >> logs, "collected %d uniq words & tags..." % (len(fullset)),

        new = new_vector() # Vector()
        for f in self.weights:

            stuff = f.split("=", 1)[1].rsplit("=", 1)[0].split("|")  ## b/w 1st and last "=", but caution
            for s in stuff:
                if s not in fullset:
                    break
            else:
                new[f] = self.weights[f]

        print >> logs, "%d features survived (ratio: %.2f)" % (len(new), len(new) / len(self.weights))
        self.weights = new

    def sparsify(self, z=1):
        '''duchi et al., 2008'''
        
        

if __name__ == "__main__":

    flags.DEFINE_string("prune", None, "prune features w.r.t. FILE (word/tag format)")

    try:
        argv = FLAGS(sys.argv)
        if FLAGS.weights is None:
            raise flags.FlagsError("must specify weights by -w ...")
    except flags.FlagsError, e:
        print >> logs, 'Error: %s\nUsage: %s ARGS\n%s' % (e, sys.argv[0], FLAGS)
        sys.exit(1)
    
    FLAGS.featstat = True
    
    model = Model(FLAGS.weights) #.model, FLAGS.weights)

    if FLAGS.prune:
        model.prune(FLAGS.prune)

    if FLAGS.outputweights:
        model.write(FLAGS.outputweights)

        
