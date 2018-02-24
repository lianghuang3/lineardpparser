#!/usr/bin/env python

''' Two classes that constitute a hypergraph (forest): Node and Hyperedge
    On top of these, there is a separate Forest class in forest.py which collects the nodes,
	and deals with the loading and dumping of forests.

	implementation details:
	1. node has a local score "node_score" and hyperedge "edge_score".
	2. and they both have "beta" \'s.

	this design is quite different from the original, where only edge-prob is present.
	
'''

import sys, os, re
import math
import copy

#sys.path.append(os.environ["NEWCODE"])

#import mycode
import heapq

logs = sys.stderr

from tree import Tree

from svector import Vector as FVector
from utility import symbol

from deptree import DepTree

class Node(Tree):
	''' Node is based on Tree so that it inherits various functions like binned_len and is_terminal. '''

	def copy(self):
		return copy.deepcopy(self)
	
	def __init__(self, iden, labelspan, size, fvector, sent):
		# NP [0-3]
		self.iden = iden
		
		label, span = labelspan.split()
		self.span = tuple(map(int, span[1:-1].split("-")))
		
		if label[-1] == "*":
			label = label[:-1]
			self._spurious = True
		else:
			self._spurious = False
			
		self.label = "TOP" if label == "S1" else label
		self.label = symbol(self.label)
		self.edges = []
		
		word = sent[self.span[0]] if (size == 0) else None
		self.prepare_stuff(label, word)

		self.fvector = fvector

		self._root = False

		self._bin_len = None


	def prepare_kbest(self):
		self.klist = []
		self.kset = set()
		self.fixed = False
		if self.is_terminal():
			self.klist = [self.bestres]
			self.kset.add(self.besttree)
			self.fixed = True
			   
		self.bestedge = None
		self.cand = None

	def mapped_span(self, mapping):
		return (mapping[self.span[0]], mapping[self.span[1]]) 
	
	def labelspan(self, separator=":", include_id=True):
		ss = "%s%s " % (self.iden, separator) if include_id else ""
		lbspn = "%s [%d-%d]" % (self.label + ("*" if self.is_spurious() else ""), \
								self.span[0], self.span[1])
		return ss + lbspn

	__str__ = labelspan
	__repr__ = __str__
	
	def is_spurious(self):
		return self._spurious

	def sp_terminal(self):
		return self.is_spurious() and self.edges[0].subs[0].is_terminal()

	def add_edge(self, hyperedge):
		self.edges.append(hyperedge)
		hyperedge.node = self ## important backpointer!

	def assemble(self, subtrees):
		'''this is nice. to be used by k-best tree generation.'''
##		t = Tree(self.label, self.span, subs=subtrees, sym=False) if not self._spurious else subtrees[0]
		if self._root:
			t = subtrees[0]
		else:
			left, right = subtrees[0], subtrees[1]
			action = (2, ) if left.headidx == int(self.label) else (1, )
			t = left.combine(right, action)
			
#			t = DepTree(int(self.label), [left], [right])
			
		assert t is not None, (self.label, self.span, subtrees, self._spurious)
		if self._root:
			## notice that, roots are spurious! so...
			pass #t.set_root(True)
		return t

	def this_tree(self):
		## very careful: sym=False! do not symbolize again
##		return Tree(self.label, self.span, wrd=self.word, sym=False)
		return DepTree(int(self.label))

	def bestparse(self, weights=FVector("0=-1"), dep=0):
		'''now returns a triple (score, tree, fvector) '''
		
		if self.bestres is not None:
			return self.bestres

		self.node_score = self.fvector.dot(weights)
		if self._terminal:
			self.beta = self.node_score
			self.besttree = self.this_tree()
			self.bestres = (self.node_score, self.besttree, FVector(self.fvector))  ## caution copy

		else:

			self.bestedge = None
			for edge in self.edges:
				## weights are attached to the forest, shared by all nodes and hyperedges
				score = edge.edge_score = edge.fvector.dot(weights)
				fvector = FVector(edge.fvector) ## N.B.! copy!
				subtrees = []
				for sub in edge.subs:
					sc, tr, fv = sub.bestparse(weights, dep+1)
					score += sc
					fvector += fv
					subtrees.append(tr)

				edge.beta = score

				if self.bestedge is None or score < self.bestedge.beta:
					self.bestedge = edge
					best_subtrees = subtrees
					best_fvector = fvector

			self.besttree = self.assemble(best_subtrees)
			self.beta = self.bestedge.beta + self.node_score
			best_fvector += self.fvector ## nodefvector

			self.bestres = (self.beta, self.besttree, best_fvector)

		return self.bestres


	def getcandidates(self, dep=0):
		self.cand = []
		for edge in self.edges:
			vecone = edge.vecone()
			edge.oldvecs = set([vecone])
			res = edge.getres(vecone, dep)
			assert res, "bad at candidates"
			self.cand.append( (res, edge, vecone) )
			
		heapq.heapify (self.cand)
		
	def lazykbest(self, k, dep=0):
		
		now = len(self.klist)
		if self.fixed or now >= k:
			return

		if self.cand is None:
			self.getcandidates(dep)
			
		while now < k:
			if self.cand == []:
				self.fixed = True
				return 
			
			(score, tree, fvector), edge, vecj = heapq.heappop(self.cand)
			if tree not in self.kset:
				## assemble dynamically
				self.klist.append ((score, tree, fvector))
				self.kset.add(tree)
				now += 1
			else:
				print >> logs, "*** duplicate %s %d" % (tree.labelspan(), now)
				
			edge.lazynext(vecj, self.cand, dep+1)
	

class Hyperedge(object):

	def unary(self):
		return not self.head.is_root() and len(self.subs) == 1

	def unary_cycle(self):
		return self.unary() and self.subs[0].label == self.head.label
	
	def __str__(self):
		return "%-17s  ->  %s " % (self.head, "  ".join([str(x) for x in self.subs]))

	def shorter(self):
		''' shorter form str: NP [3-5] -> DT [3-4]   NN [4-5]'''
		return "%s  ->  %s " % (self.head.labelspan(include_id=False), \
								"  ".join([x.labelspan(include_id=False) for x in self.subs]))			

	def shortest(self):
		''' shortest form str: NP -> DT NN '''
		return "%s  ->  %s " % (self.head.label, "  ".join([str(x.label) for x in self.subs]))			
							 
	__repr__ = __str__

	def __init__(self, head, tails, fvector):
		self.head = head
		self.subs = tails
		self.fvector = fvector

	def arity(self):
		return len(self.subs)

	def vecone(self):
		return (0,) * self.arity()

	def compatible(self, tree, care_POS=False):
		if self.arity() == tree.arity():
			
			for sub, tsub in zip(self.subs, tree.subs):
				if not sub.compatible(tsub, care_POS):
					return False
			return True

	def getres(self, vecj, dep=0):
		score = self.edge_score 
		fvector = self.fvector + self.head.fvector
		subtrees = []
		for i, sub in enumerate(self.subs):

			if vecj[i] >= len(sub.klist) and not sub.fixed:
				sub.lazykbest(vecj[i]+1, dep+1)
			if vecj[i] >= len(sub.klist):
				return None
			
			sc, tr, fv = sub.klist[vecj[i]]
			subtrees.append(tr)
			score += sc
			fvector += fv
		
		return (score, self.head.assemble(subtrees), fvector)

	def lazynext(self, vecj, cand, dep=0):
		for i in xrange(self.arity()):
			## vecj' = vecj + b^i (just change the i^th dimension
			newvecj = vecj[:i] + (vecj[i]+1,) + vecj[i+1:]

			if newvecj not in self.oldvecs:
				newres = self.getres(newvecj, dep)
				if newres is not None:
					self.oldvecs.add (newvecj)
					heapq.heappush(cand, (newres, self, newvecj))

	@staticmethod
	def _deriv2tree(edgelist, i=0):
		'''convert a derivation (a list of edges) to a tree, using assemble
		   like Tree.parse, returns (pos, tree) pair
		'''
		edge = edgelist[i]
		node = edge.head
		subs = []
		for sub in edge.subs:
			if not sub.is_terminal():
				i, subtree = Hyperedge._deriv2tree(edgelist, i+1)
			else:
				subtree = sub.this_tree()
			subs.append(subtree)

		return i, node.assemble(subs)

	@staticmethod
	def deriv2tree(edgelist):
		_, tree = Hyperedge._deriv2tree(edgelist)
		return tree
	
	@staticmethod
	def deriv2fvector(edgelist):
		'''be careful -- not only edge fvectors, but also node fvectors, including terminals'''
		
		fv = FVector()
		for edge in edgelist:
			fv += edge.fvector + edge.head.fvector
			for sub in edge.subs:
				if sub.is_terminal():
					fv += sub.fvector
		return fv
