#!/usr/bin/env python2

import sys
from os import path
import numpy as np

# assumes l is sorted ascending (low to high)
# Returns True if value is in list l, False otherwise
def binarySearchFor(value, l):
  if len(l) == 0: # l is empty
    return False
  
  pivot = len(l)/2
  pivotVal = l[pivot]

  if value < pivotVal:
    return binarySearchFor(value, l[:pivot]) # search left

  if value > pivotVal:
    return binarySearchFor(value, l[(pivot+1):]) # search right

  return True # value == pivotVal

if __name__ == '__main__':

  if len(sys.argv) != 4:
    sys.stderr.write("Usage: python2 corpus.py CORPUSFILE DESTDIR RATIO \n")
    sys.stderr.write("  e.g. `python2 corpus.py data/sans/TaggedCorpus.txt data/sans/ 0.2` creates a sans/train.txt and sans/test.txt, with the test 20% of CORPUS and the train 80% of CORPUS")
    sys.exit(1)

  fname = sys.argv[1] # tagged corpus
  destdir = sys.argv[2] # destination directory
  r = float(sys.argv[3]) # ratio

  f = open(fname, "r") # open the corpus for reading
  trainF = open(path.join(destdir, "train.txt"), "w")
  testF = open(path.join(destdir, "test.txt"), "w")

  corpus = [l for l in f]

  N = len(corpus)
  test_size = int(N*r) # number of lines in test corpus
  lines = np.random.random_integers(low=0,high=N-1,size=test_size)

  for i in lines:
    testF.write(corpus[i])

  lines.sort()

  for i in xrange(0,N):
    if not binarySearchFor(i, lines): # i cannot be in list of indices used for test
      trainF.write(corpus[i])

  f.close()
  trainF.close()
  testF.close()

