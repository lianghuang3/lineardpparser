Linear-Time Dynamic Programming Shift-Reduce Dependency Parser.
===

This parser is described in the following two papers:

1. Liang Huang and Kenji Sagae (2010).
   Dynamic Programming for Linear-Time Incremental Parsing.
   Proceedings of ACL.

2. Liang Huang, Suphan Fayong, and Yang Guo (2012).
   Structured Perceptron with Inexact Search.
   Proceedings of NAACL.

Note that the default trainer is the max-violation perceptron in paper 2,
so is the latest model included. 

The trainer also implements the parallized perceptron of McDonald et al (2010).

Author: Liang Huang <liang.huang.sh@gmail.com>

User Manual.

This package requires Python 2.7.

0 DIRECTORIES AND INSTALLATION 
====================

  code/	      parser code
  data/	      English Treebank, tagged by jackknifing
  models/     trained models
  hvector/    sparse vector class (included as a bonus)

  The code should be runnable as is on most 64-bit Linux platforms.

  On other platforms (e.g. Mac OS X or 32-bit Linux, or if it doesn't run), try:
  
	cd code; python setup.py install --install-lib .

  This would recompile the sparse vector libraries for your platform.
  This step requires gcc and Python header files. If the latter are missing, try something like
  
        sudo apt-get install python-dev
  
  or get them from python.org (just download python source code which contains header files).

1 PARSING 
=========================================

	Usage:
		cat <input_file> | ./code/parser.py -w <model_file> [-b <beam_width>]

	beam_width is 8 by default.


	a) to parse POS-tagged sentences:

		echo "I/PRP am/VBP a/DT scientist/NN ./." | ./code/parser.py -w models/small_model.b1 -b1

	output would be something like:
		((I/PRP) am/VBP ((a/DT) scientist/NN) (./.))
		sentence 1    (len 5): modelcost= 119.94	prec= 100.00	states= 9 (uniq 9)	edges= 18	time= 0.00


	b) to parse and evaluate accuracy against gold trees:
	
		echo "((I/PRP) am/VBP ((a/DT) scientist/NN) (./.))" | ./code/parser.py -w ...

	output would be something like:
		((I/PRP) am/VBP ((a/DT) scientist/NN) (./.))
		sentence 1    (len 5): modelcost= 263.64	prec= 100.00	states= 71 (uniq 71)	edges= 133	time= 0.04


	c) interactive mode (simply no input):

		./code/parser.py -w ...


	d) other parameters:

		./code/parser.py:
		  --forest: dump dependency forest
		  -k, --kbest: k-best output

	e) to replicate Huang et al (2012) results:
	
		cat data/23.dep.autotag | ./code/parser.py -w models/modelptb.max.b8 -b8 \
                                        > 23.parsed

 	   the results should be identical to data/23.dep.parsed, and the logs should be 
	   comparable to data/23.dep.parsed.log. 
	   
	   Note that loading the big model is extremeley slow (~1 min), but parsing should
	   be very fast (~0.04 seconds per sentence).

2 TRAINING  
=======================================

  Usage:
     ./code/multitrainer.py --train <train> --dev <dev> -i <iter> \
     			    --out <trained_model> -w <templates> \
			    -b<beam> [--update <method>] [--multi <multi>] 
  
  iter: number of iterations or "epochs" (15 by default).
  update: max (default), early, and noearly (std perc, terrible performance).
  multi: parallelized perceptron a la iterative parameter mixing of McDonald et al (2010).

  finaldump: hold the best weights and dump at the end; might save a little bit of time but requires 1.5 times memory (by default, dump at each new high)
			   
  * To replicate Huang et al (2012) results (max violation update):

        ./code/multitrainer.py --train data/02-21.dep.4way --dev data/22.dep.4way \
  			 --out mymodel -w models/q2.templates -b8 -i15 2>logfile

    This would take about 5.6 hours on a Xeon E5-2687W CPU @3.1 GHz, using 1 GB of memory. 
    The output model should be identical to models/model.max.b8. 
    The log file should print among others:

        ... example 1000 (len 21)... sim=0.00%, |delta|=667
        ... example 2000 (len 20)... sim=0.00%, |delta|=756
        ...
        
    To extract per-epoch information you can "grep ^at logfile", which should print:
    
        at iter 1, updates 39832 (early 32334, er 81.2%), dev 90.62%+, |w| 3846296, ...
        at iter 2, updates 39832 (early 31297, er 78.6%), dev 91.49%+, |w| 4950067, ...
        ... 
        at iter 12, updates 39832 (early 22453, er 56.4%), dev 92.32%+, |w| 7460262, ...
        ...

    where a "+" sign indicates a new highest score on dev, and |w| is the model size.
    A sample log file is provided in models/log.max.b8 (iter 12 is the highest score).

  * To replicate Huang and Sagae (2010) results (early update):

    Just use "--update early" and "-i40" (early update converges a lot slower!):

        ./code/multitrainer.py --train data/02-21.dep.4way --dev data/22.dep.4way \
  			 --out mymodel.early -w models/q2.templates -b8 \
             -i40 --update early 2>logfile

    Note initially each epoch will be a lot faster than max violation due to early update.

        at iter 1, updates 32220 (early 31198, er 96.8%), dev 89.21%+, |w| 1872282,
        at iter 2, updates 28483 (early 27366, er 96.1%), dev 90.30%+, |w| 2794786,
        ...

    This would take about 13 hours on the above-mentioned CPU, using 1 GB of memory.

  * To use parallelized perceptron of McDonald et al (2010):

    Use --multi <m> (e.g., m=10) and a much larger -i <i>. 
    Each epoch will be roughly speaking m times faster, but the accuracies suffer a lot;
    so you need many more epochs than before (e.g., -i70 for max and -i300 for early).
    In the end the speedup would be around 1.5 (with m=10).
    (Note McDonald et al (2010) report speedups around 3 with m=10, 
     but they use standard update with exact search).
    If m is too small (<8) or too big (>20), the speedup might be even worse.
    In fact it is quite possible to see no speedup at all or even slower than serial.

    Also note big m requires more memory (about 2m x).

    Adding --shuffle would often improve the results, but no longer deterministic.

3 FEATURES TO COME VERY SOON 
=====================

  * parallelized parsing and model reading.
    (currently supports parallelized training and evaluation on dev, 
	using parallelized perceptron of McDonald et al 2010)

  * minibatch parallelized percpetron training (Zhao and Huang, 2013),
	much faster than McDonald et al (2010).
# lineardpparser
