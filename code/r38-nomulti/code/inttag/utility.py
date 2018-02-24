#!/usr/bin/env python

''' utility functions used by features.py
'''

import sys

def quantize(v):
	if v <= 2:
		return v
	elif v <= 4:
		return 4
	return 5

def bound_by(v, m):
	if v >= m:
		return m
	if v <= -m:
		return -m
	return v

def aset(st):
	''' aset("A B C") => set(["A", "B", "C"])'''
	return set(st.split())

def symbol(s):
	return s.replace("*", "\*").replace("%", "\%").replace("/", "\/")

def desymbol(s):
	return s.replace("\*", "*").replace("\%", "%").replace("\/", "/").replace('\\"', '"').replace("@UNKNOWN@", "")  ## \"=" regardless

punc_tags = aset("'' : # , . `` -LRB- -RRB-")

conj_tags = aset("CC CONJP")  # CONJP is NT: (CONJP (CC but) (not but))   (CONJP (RB rather) (IN than))
## confusion: in Johnson's code, is_conj means is_term and CC, so why include CONJP here?

func_tags = aset("CC DT EX IN MD POS PRP PRP$ RP TO WDT WP WP$")

def is_punc(label):
	return label in punc_tags

def is_conj(label):
	return label in conj_tags

def is_func(label):
	return label in func_tags

def make_punc((tag, word)):
	'''should always use POS tag to determ if is punc'''
	return word if is_punc(tag) else "_"

def words_from_line(line):
	''' return the number of tokens in a PTB-style parse tree. (fast) '''
	stuff = line.split()
	return [x[:x.find(")")] for x in stuff if x[-1]==')']

def num_words(line):
	return len(words_from_line(line))

def getfile(name, oper=0):

	tag = ["r", "w", "a"][oper]
	default = [sys.stdin, sys.stdout, sys.stderr][oper]

	if type(name) is str:  
		if name == "-":
			name = default
		else:
			name = open(name, tag)
	return name

def xzip(a, b):
	'''lazy zip. make sure both arguments are lazy!'''

	for x in a:
		try:
			y = b.next()
			ret = ((x, y))
			yield ret
		except:
			break
	

######### utf tools ##############

encodings = ["ascii", "utf-8", "gb2312"]

def detect_encoding(word):
	" detect and decode; return (encoding, decoded_str) pair "

	for enc in encodings:
		try:
			chars = word.decode(enc)
			return enc, chars
		except:
			pass

	raise Exception("Unknown Encoding for %s" % word)
	
def words_to_chars(wstr, split_ascii=False, encode_back=False):
	" return a list of chars "

	if type(wstr) is str:
		wstr = wstr.split()

	chars = []
	for word in wstr:
		enc, cs = detect_encoding(word)

		if enc == "ascii" and not split_ascii:
			if encode_back:
				cs = cs.encode(enc)
			chars.append(cs)
		else:
			if encode_back:
				cs = [x.encode(enc) for x in cs]
			chars.extend(cs)

	return chars
