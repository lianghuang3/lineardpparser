#!/usr/bin/env python

''' A Feature Vector (Id -> value mapping)
'''

import sys
import copy
import math

from utility import getfile

logs = sys.stderr
print_features = False

precision = 1e-4

def pp_id_v((Id, v)):

	if Id == 0:
##		return "%d=%.4lf  " % (Id, -v)  ## negative!! same as read-in
		return "%d=%.4lf  " % (Id, -v)
	else:
##		return ("%7d    " % Id if v == 1 else "%7d=%-3d" % (Id, v))
		return ("%d" % Id if v == 1 else "%d=%d" % (Id, v) if type(v) is int else "%d=%.4lf" % (Id, v))

def pp_id_v_name((Id, v)):

	name = FVector.reverse_mapping[Id]
	if type(v) is float:
		return "%s=%.4lf  " % (name, -v)
	else:
		return ("%s" % name if v == 1 else "%s=%-d" % (name, v))

class FVector(dict):
	''' TODO: special care for unit-valued features.
	    also modelling weight vectors.
	'''

		
	def copy(self):
		'''shallow-copy'''

		return copy.copy(self)
	
	def dot_product(self, weights):
		sc = 0
		
		for Id in self:
			w = weights.get(Id, 0)  ## do NOT put a zero here!
			sc += self[Id] * w
			
		return sc

	__mul__ = dot_product

	def __rmul__(self, scale):
		new = {}
		for Id in self:
			new[Id] = self[Id] * scale
		return FVector(new)

	def __div__(self, scale):
		scale = float(scale)
		new = {}
		for Id in self:
			new[Id] = self[Id] / scale
		return FVector(new)

	def __imul__(self, scale):
		if math.fabs(scale - 1) > precision:			
			for Id in self:
				self[Id] = self[Id] * scale
		return self

	def norm2(self):
		return self * self

	def norm(self):
		return math.sqrt(self.norm2())

	def __iadd__(self, fvector):

		for Id in fvector:
			if Id not in self:
				self[Id] = fvector[Id]
			else:
				self[Id] += fvector[Id]

		return self

	def __isub__(self, fvector):

		for Id in fvector:
			if Id not in self:
				self[Id] =- fvector[Id]
			else:
				self[Id] -= fvector[Id]

			if math.fabs(self[Id]) < precision:
 				del self[Id]

		return self

	def __sub__(self, fvector):

		new = self.copy()
		for Id in fvector:
			if Id not in self:
				new[Id] =- fvector[Id]
			else:
				new[Id] -= fvector[Id]

			if math.fabs(new[Id]) < precision:
 				del new[Id]

		return new

	def __add__(self, fvector):

		new = self.copy()
		for Id in fvector:
			if Id not in self:
				new[Id] = fvector[Id]
			else:
				new[Id] += fvector[Id]

		return new

##	def __sub__(self, fvector):

##		delta = {}
##		keys = set(self.keys()) | set(fvector.keys())
##		for Id in keys:
##			d = self.get(Id, 0) - fvector.get(Id, 0)  ## try to be sparse
##			if d != 0:
##				delta [Id] = d

##		return FVector(delta)
		


	def pp(self, sentid=0, cross_lines=False, usename=False):
		
		toprint = []
		for Id, v in self.items():
			toprint.append((Id, v))
		
		toprint.sort()
		if cross_lines:
			return "\n".join(["#%d\t%s" % (sentid, pp_id_v(id_v)) for id_v in toprint]) + "\n"
		else:
			if usename:
				return (("#%d" % sentid) if sentid !=0 else "") + \
					   "\t%s" % " ".join([pp_id_v_name(id_v) for id_v in toprint])
			else:
				return (("#%d" % sentid) if sentid !=0 else "") + \
					   "\t%s" % " ".join([pp_id_v(id_v) for id_v in toprint])

	def __str__(self):
		#return " ".join([pp_id_v(idv) for idv in self.items()])
		return self.pp()

	def error(self, other, details=False):
		'''report difference and quit'''
		diff = self - other
		s = "diff=%s" % diff
		if details:
			s += "\nnew =%s\nold =%s" % (self, other)
		assert False, s
		
	def check(self, other, details=False):
		''' like __cmp__, but i do not want to override it here for efficiency considerations.'''

		if math.fabs(self[0] - other[0]) > 1e-4:
			self.error(other, details)
		tmp = self[0]
		self[0] = other[0]
		if self != other:
			self.error(other, details)
		self[0] = tmp
		return True
		
	@staticmethod
	def parse(fields):
		fvector = FVector()
		for i, field in enumerate(fields.split()):
			idval = field.split("=")
			if len(idval) == 2:
				Id, val = int(idval[0]), float(idval[1])
				if math.fabs(val - int(val)) < 1e-4:
					val = int(val)
				if Id==0:
					val = -val ## special
			else:
				try:
					Id = int(field)
					### VERY VERY CAREFUL HERE: case like -112
					if Id > 0:
						val = 1
					else:
						raise Exception
				except:
					Id = 0  
					val = -float(field)   # logprob
				
			assert Id not in fvector, "duplicate feature: %d in %s" % (Id, fields)
			fvector[Id] = val

		return fvector

	@staticmethod
	def convert_fullname(fs):

		fvector = {}
		for f in fs:
			if print_features:
				print >> logs, f

			Id = FVector.feature_mapping.get(f, -1)

			if Id > -1:
				fvector[Id] = fvector.get(Id, 0) + 1

##		for f, v in fs:
##			if print_features:
##				print >> logs, f
##			if f in FVector.feature_mapping:
##				## a known feature

##				Id = FVector.feature_mapping[f]
##				if Id not in fvector:
##					fvector[Id] = 0

##				fvector[Id] += v

		return FVector(fvector)

	@staticmethod
	def readweights(filename):
		'''read the first line only. weights must start with a "W" '''
		line = getfile(filename).readline()
		if line[0] == "W":
			return FVector.parse(line[1:])
		return None
	
