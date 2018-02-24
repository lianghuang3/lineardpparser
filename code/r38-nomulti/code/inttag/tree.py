#!/usr/bin/env python

'''A PTB style tree for reranking
   N.B.: The biggest difference between my design and that of Johnson is
   that I do not have a preterminal level. the leaves of a tree have both POS and word.
   Be careful when you translate Johnson code into Python!

   Also note that my code can automatically remove empty nodes (*NONE*) and traces (NP-1) and stuff...
'''

import re, sys
logs = sys.stderr

from utility import quantize, aset, symbol, desymbol, is_conj, is_punc, is_func

import heads
htypes = [heads.SEM, heads.SYN]

print_heads = False

class Tree(object):

	def prepare_stuff(self, label, wrd=None, sym=True):
		
		self._coordination = None ## to be evaluated once called (same as C++'s const)
		self._str = None
		## heads-info
		self.headinfo = { heads.SEM: heads.HeadInfo(), heads.SYN: heads.HeadInfo()}

		if wrd is not None:
			self.word = symbol(wrd) if sym else wrd

			self._terminal = True
			self._punctuation = is_punc(self.label)
			self._conjunction = is_conj(self.label)

			self.word_seq = [self.word]
			self.tag_seq = [label]
			
		else:
			self._terminal = False
			self._punctuation = False
			self._conjunction = False			

			self.word_seq = []
			self.tag_seq = []			

	def __init__(self, label, span, wrd=None, subs=None, is_root=False, sym=True):

		self.parentlabel = None ### TODO: FIX THIS!

		self.label = symbol(label) if sym else label ## in forest.assemble, don't symbol again
		self.span = span
		assert (wrd is None) ^ (subs is None), "bad tree"

		self.prepare_stuff(label, wrd, sym)

		if not self._terminal:
			self.subs = subs
			for sub in subs:
				self.word_seq += sub.word_seq
				self.tag_seq += sub.tag_seq
				
		self._root = is_root

		## features
		self._bin_len = None


		# for heads feature
		self.allheads = {}
		self.ccheads = {}
		self.twolevels = {}

		# for headtree feature
		self.headspath = {}

	def get_sent(self):
		return self.word_seq

	def get_tags(self):
		return self.tag_seq

	def is_terminal(self):
		return self._terminal

	def is_spurious(self):
		''' by default there is no spurious level in a normal tree.
		    will be overridden in forest nodes.
		'''
		return False

	def sp_terminal(self):
		return self.is_terminal() or self.is_spurious() and self.subs[0].is_terminal()

	def is_punctuation(self):
		return self._punctuation

	def is_conjunction(self):
		return self._conjunction

##  is_coordination() is true iff one of the non-first, non-last children 
##  is a conjunction (CC) (the first and last child are ignored so constructions 
##  beginning with a conjunction don't count as coordinated).
	
	def _is_coordination(self):
		if self.is_terminal(): # or len(self.subs) <3: [1:-1] already makes sure this
			return False
		for sub in self.subs[1:-1]:  ## miao! exercise for students to show no need for range
			if sub.is_conjunction():
				return True
		return False
		
	def is_coordination(self):
		if self._coordination is None:
			self._coordination = self._is_coordination()

		return self._coordination

	def is_root(self):
		return self._root

	def set_root(self, is_root):
		self._root = is_root
	
	def dostr(self):
		if self.is_terminal():
#			print "hi", self.word, desymbol(self.word)
#			print "ho", self.label, desymbol(self.label)
			s = "(%s %s)" % (self.label, self.word)
		else:
#			print "ho", self.label, desymbol(self.label)
			s = "(%s %s)" % (self.label, " ".join([str(sub) for sub in self.subs]))

		if self.is_root(): ## only once! DON'T DO IT RECURSIVELY!
			s = desymbol(s)
		self._hash = hash(s)
		return s

	def cased_str(self, cased_sent):
		''' normal case '''
		if self.is_terminal():
#			print "hi", self.word, desymbol(self.word)
#			print "ho", self.label, desymbol(self.label)
			s = "(%s %s)" % (self.label, cased_sent[self.span[0]])
		else:
#			print "ho", self.label, desymbol(self.label)
			s = "(%s %s)" % (self.label, \
							 " ".join([sub.cased_str(cased_sent) for sub in self.subs]))
			
		# don't desymbol again!	dostr already desymboled.
# 		if self.is_root(): ## only once! DON'T DO IT RECURSIVELY!
# 			s = desymbol(s)
		return s

	def rehash(self):
		'''when the children nodes are re-set, recompute relevant info. used by local_feats.py'''
		self._coordination = None
		self._str = None
		self.is_coordination()
		self.__str__()

		for htype in htypes:
			if self.is_terminal():
				self.set_headchild(htype, self)				
			else:
				self.set_headchild(htype, heads.headchild(htype, self))
		
	def __str__(self):
		if self._str is None:
			self._str = self.dostr()
		return self._str

	__repr__ = __str__

	def __hash__(self):
		if self._str is None:
			self._str = self.dostr()
		return self._hash

	def __eq__(self, other):
		### CAUTION!
		return str(self) == str(other)
		### return id(self) == id(other)

	def equal(self, other, care_pos=False):
		if self.is_terminal():
			return other.is_terminal()
		else:
			if not other.is_terminal() and self.label == other.label and self.span == other.span:
				for sub, osub in zip(self.subs, other.subs):
					if not sub.equal(osub, care_pos):
						return False
				return True
			return False

	def span_width(self):
		return self.span[1] - self.span[0]

	def binned_len(self):
		if self._bin_len is None:
			self._bin_len = quantize(self.span_width())
		return self._bin_len

	__len__ = span_width		

	def arity(self):
		return len(self.subs)

	def labelspan(self):
		return "%s [%d-%d]" % (self.label, self.span[0], self.span[1])

	def spanlabel(self):
		return "[%d-%d]: %s" % (self.span[0], self.span[1], self.label)

	def get_headchild(self, htype):
		return self.headinfo[htype].headchild

	def get_headchild_pos(self, htype):
		head = self.headinfo[htype].headchild
		if self.is_terminal():
			return 0
		else:
			for i, sub in enumerate(self.subs):
				if sub is head:
					return i

	def get_lexhead(self, htype):
		return self.headinfo[htype].lexhead

	def set_headchild(self, htype, headchild):
		self.headinfo[htype].headchild = headchild

	def set_lexhead(self, htype, lexhead):
		self.headinfo[htype].lexhead = lexhead

	def set_max_outs(self, max_outs):
		for htype in htypes:
			self.headinfo[htype].max_out = max_outs[htype]

	def get_maximal_outside(self, htype):
		p = self.headinfo[htype].max_out
		return map(lambda x:x.label, p)

	def get_tag_word(self, index):
		return (self.tag_seq[index], self.word_seq[index])

	def annotate(self, max_outs, parentlabel=None, do_sub=True, dep=0):
		''' this is TOP-DOWN annotation.
		    max-outs = { SYN: (max, out), SEM: (max1, out1) }
		'''

		self.parentlabel = parentlabel
		if max_outs is not None:
			self.set_max_outs(max_outs)		
			
		if self.is_terminal():
			for htype in htypes:
				self.set_lexhead(htype, heads.LexHead(self))
				self.set_headchild(htype, self)

		else:
			for htype in htypes:
				self.set_headchild(htype, heads.headchild(htype, self))
##				print >> logs, " " * dep, self.labelspan(), htype, heads.headchild(htype, self)

			if do_sub:
				headss = {}
				for htype in htypes:
					headss[htype] = self.get_headchild(htype)
				for sub in self.subs:
					if max_outs is not None:
						new_max_outs = {}
						for htype in htypes:
							### SUPER CAUTION! is, not ==; otherwise parallels like "rally, rally, rally"
							if sub is headss[htype]:
								## i am head, so adopt parent's max_outs
								new_max_outs [htype] = max_outs [htype]
							else:
								## head changes at this level
								new_max_outs [htype] = (sub, self)
					else:
						new_max_outs = None

					sub.annotate(new_max_outs, self.label, dep+1)					
			
			for htype in htypes:
				self.set_lexhead(htype, self.get_headchild(htype).get_lexhead(htype))
				
				if print_heads:
					print " |  " * dep, self.spanlabel(), " ", self.get_lexhead(htype)


	def annotate_all(self):
		assert(self.is_root())
		realroot = self.subs[0]  # realroot = "S", self = "S1"
		import heads
		realroot.annotate({heads.SEM: (realroot, self), heads.SYN: (realroot, self)}, self.label)
		self.set_headchild(heads.SEM, realroot)
		self.set_headchild(heads.SYN, realroot)
		self.word_seq = realroot.word_seq
		self.parentlabel = None

	@staticmethod
	def parse(line, trunc=True, lower=False, annotate=False):

		_, is_empty, tree = Tree._parse(line, 0, 0, trunc, lower)

		assert not is_empty, "The whole tree is empty! " + line

		tree.label = "TOP"

		if annotate:
			# recursive annotations (heads and stuff)
			tree.annotate_all()

		return tree			
		
	@staticmethod
	def _parse(line, pos=0, wrdidx=0, trunc=True, lower=False):
		''' returns a triple:
		    ( (pos, wordindex), is_empty, tree)
			The is_empty bool tag is used for eliminating emtpy nodes recursively.
			Note that in preorder traversal, as long as the indices do not advance for empty nodes,
			it is fine for stuff after the empty nodes.
		'''
		## (S1 (S (ADVP (RB No)) (, ,) (NP (PRP it)) (VP (VBD was) (RB n't) (NP (JJ Black) (NNP Monday))) (. .)))
		assert line[pos]=='(', "tree must starts with a ( ! line=%s, pos=%d, line[pos]=%s" % (line, pos, line[pos])
			
		empty = False
		space = line.find(" ", pos)
		label = line[pos + 1 : space]
		if trunc:
			## remove the PRDs from NP-PRD
			if label[0] != "-":
				dashpos = label.find("-")			
				if dashpos >= 0:
					label = label[:dashpos]

				## also NP=2 coreference (there is NP-2 before)
				dashpos = label.find("=")			
				if dashpos >= 0:
					label = label[:dashpos]

				## also ADVP|PRT and PRT|ADVP (terrible!)
				dashpos = label.find("|")			
				if dashpos >= 0:
					label = label[:dashpos]

			else:
				## remove traces
				## CAUTION: emptiness is recursive: (NP (-NONE- *T*-1))
				if label == "-NONE-":
					empty = True
				
		newpos = space + 1
		newidx = wrdidx
		if line[newpos] == '(':
			## I am non-terminal
			subtrees = []			
			while line[newpos] != ')':
				if line[newpos] == " ":
					newpos += 1
				(newpos, newidx), emp, sub = Tree._parse(line, newpos, newidx, trunc, lower)
				if not emp:
					subtrees.append(sub)
				
			return (newpos + 1, newidx), subtrees==[], \
				   Tree(label, (wrdidx, newidx), subs=subtrees, is_root=(pos==0))
		
		else:
			## terminal
			finalpos = line.find(")", newpos)
			word = line[newpos : finalpos]
			if lower:
				word = word.lower()
			## n.b.: traces shouldn't adv index!
			return (finalpos + 1, wrdidx + 1 if not empty else wrdidx), empty, Tree(label, (wrdidx, wrdidx+1), wrd=word)  
		


	def all_label_spans(self):
		'''get a list of all labeled spans for PARSEVAL'''

		if self.is_terminal():
			return []
		
		a = [(self.label, self.span)]
		for sub in self.subs:
			a.extend(sub.all_label_spans())

		return a

	def is_same_sentence(self, other):
		##assert other isinstance(other, Tree), "the other tree is not of Tree type"

		return len(self) == len(other) and self.get_sent() == other.get_sent()		
		
			
###########################################

## attached code does empty node removal ##

###########################################

if __name__ == "__main__":

	max_len = 400
	if len(sys.argv) > 1 and sys.argv[1].find("-c") >= 0:
		max_len = int(sys.argv[1][2:])
		del sys.argv[1]

	for line in sys.stdin:
		t = Tree.parse(line.strip(), lower=False)
		if len(t) <= max_len:
			print t
