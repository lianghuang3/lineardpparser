#!/usr/bin/env python

'''finding syntactic and semantic heads. after Johnson heads.h and heads.cc'''

## needs heavy annotation -- esp. left/right head-finding.

import sys
import copy

#from tree import Tree

logs = sys.stderr

from utility import aset, is_punc

SEM = 0
SYN = 1
headtables = { SEM: None, SYN: None }

class LexHead(object):

	__slots__ = "label", "word"

	def __getstate__(self):
		return None

	def __init__(self, tree):
		self.label = tree.label
		self.word = tree.word

	def __str__(self):
		return "(%s %s)" % (self.label, self.word)

	__repr__ = __str__

	def is_punctuation(self):
		return is_punc(self.label)

class HeadInfo(object):

	def __getstate__(self):
		return None

	__slots__ = "lexhead", "headchild", "max_out"  # max-out=(maximal, outside)

	def __init__(self, lexhead=None, headchild=None, max_out=None):
		self.lexhead = lexhead
		self.headchild = headchild
		self.max_out = max_out

	def __str__(self):
		return self.lexhead.__str__()

	__repr__ = __str__

	def copy(self):
		return copy.copy(self)

class HeadTable(object):

	def __getstate__(self):
		return None

	__slots__ = "rightheaded_nominals", \
				"adjective", "conjunction", "interjection", "noun", "preposition", "verb", \
				"unknown", \
				"headtype", "table_name"

def set_headtype(this):

	## this assignment is identical for syn and sem

	this.headtype = { "ADJP" : this.adjective, 
					  "ADVP" : this.verb,
					  "CONJP" : this.conjunction,
					  "FRAG" : this.noun,
					  "INTJ" : this.interjection,
					  "LST" : this.noun,
					  "NAC" : this.noun,
					  "NP" : this.noun,
					  "NX" : this.noun,
					  "PP" : this.preposition,
					  "PRN" : this.noun,
					  "PRT" : this.preposition,
					  "QP" : this.noun,
					  "ROOT" : this.verb,
					  "RRC" : this.verb,
					  "S" : this.verb,
					  "SBAR" : this.verb,
					  "SBARQ" : this.verb,
					  "SINV" : this.verb,
					  "SQ" : this.verb,
					  "S1" : this.verb,
					  ## N.B. S1 -> TOP
					  "TOP" : this.verb,
					  "UCP" : this.adjective,
					  "VP" : this.verb,
					  "WHADJP" : this.adjective,
					  "WHADVP" : this.adjective,
					  "WHNP" : this.noun,
					  "WHPP" : this.preposition,
					  "X" : this.unknown }


def init_syn():
	if headtables[SYN] is not None:
		return

	this = HeadTable()
	headtables[SYN] = this

	this.table_name = "SYN"

	this.rightheaded_nominals = aset("NN NNS NNP NNPS $")

	this.unknown = []

	this.adjective = [aset("$ CD JJ JJR JJS RB RBR RBS WRB"), \
					  aset("ADJP ADVP")]

	this.conjunction = [aset("CC")]
	this.interjection = [aset("INTJ UH")]

	this.noun = [aset("POS"), \
				 aset("DT WDT WP$ WP PRP EX"), \
				 aset("NN NNS"), \
				 aset("$ NNP NNPS"), \
				 aset("-NONE- QP NP NP$ WHNP"), \
				 aset("CD IN JJ JJR JJS PDT RB PP")]

	this.preposition = [aset("IN RP TO"), \
						aset("PP")]

	this.verb = [aset("AUX AUXG MD"), \
				 aset("VB VBD VBG VBN VBP VBZ"), \
				 aset("VP"), \
				 aset("ADJP JJ S SINV SQ TO")]

	set_headtype(this)


def init_sem():
	if headtables[SEM] is not None:
		return

	this = HeadTable()
	headtables[SEM] = this

	this.table_name = "SEM"

	this.rightheaded_nominals = aset("")  # no such thing for semantic

	this.unknown = []


	this.adjective = [aset("$ CD JJ JJR JJS RB RBR RBS WRB"), \
					  aset("ADJP ADVP")]

	this.conjunction = [aset("CC")]
	this.interjection = [aset("INTJ UH")]

	this.noun = [aset("EX NN NNS PRP WP"), \
				 aset("$ NNP NNPS"), \
				 aset("QP NP WP$"), \
				 aset("CD DT IN JJ JJR JJS PDT POS RB WDT")]

	this.preposition = [aset("IN RP TO"), \
						aset("PP")]

	this.verb = [aset("VP"), \
				 aset("VB VBD VBG VBN VBP VBZ"), \
				 aset("ADJP JJ S SINV SQ TO"), \
				 aset("AUX AUXG MD")]

	set_headtype(this)

##############################################

init_syn()
init_sem()



def headchild(htype, t):
	if t.is_terminal():
		return t

##	print >> logs, "finding head for", t.spanlabel()

	label = t.label
	head = None

	headtable = headtables[htype]
	if label in headtable.headtype:

		headtype = headtable.headtype[label]

		for cats in headtype:
			for child in t.subs:
				if child.label in cats:
					head = child
					if headtype == headtable.verb or headtype == headtable.preposition \
					   or (htype == SYN and headtype == headtable.noun
						   and child.label not in headtable.rightheaded_nominals):
						## caution: last clause only for SYN!
						break
			if head is not None:
				return head

	# didn't find a head; return right-most non-punctuation preterminal

	revsubs = t.subs[:]
	revsubs.reverse()

	for child in revsubs:
		if child.is_terminal() and not child.is_punctuation():
			return child

	# still no head -- return right-most non-punctuation

	for child in revsubs:
		if not child.is_punctuation():
			return child

	return t.subs[0]
##	assert False, "no %s head found for %s" % (htype, t)

##	print >> logs, "head label %s not found in table %s" % (label, headtable.table_name)
