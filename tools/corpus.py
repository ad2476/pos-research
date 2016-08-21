#!/usr/bin/env python2

import sys
import argparse
from os import path
import numpy as np

def parseProgramArgs():
  parser = argparse.ArgumentParser(description="Split a corpus into test/train documents.")
  parser.add_argument("corpus", help="Path to corpus")
  parser.add_argument("output_dir", help="Directory into which to output train.txt and test.txt")
  mutexgroup = parser.add_mutually_exclusive_group(required=True)
  mutexgroup.add_argument("-l", "--line", type=int, help="Line number in corpus to use as test (for cross-validation)")
  mutexgroup.add_argument("-r", "--ratio", type=float, help="Ratio of corpus to use as test (random subset)")
  mutexgroup.add_argument("-n", "--size", type=int, help="Size of test corpus, by line count (random subset)")

  return parser.parse_args()

if __name__ == '__main__':
  args = parseProgramArgs()

  fname = args.corpus
  destdir = args.output_dir

  f = open(fname, "r") # open the corpus for reading
  trainF = open(path.join(destdir, "train.txt"), "w")
  testF = open(path.join(destdir, "test.txt"), "w")

  corpus = [l for l in f]

  N = len(corpus)
  if args.size or args.ratio: # Want to randomly select some number of lines into test/train:
    if args.size:
      test_size = args.size # this is how many lines we want in the test corpus
    else: # otherwise, we're going by ratio:
      test_size = int(N*args.ratio) # number of lines in test corpus

    # randomly select $test_size line numbers in range 0,N-1
    lines = np.random.random_integers(low=0,high=N-1,size=test_size)
    for i in lines: # use these for the test corpus:
      testF.write(corpus[i])

    # next, put the rest of the lines in the train corpus:
    for i in xrange(0,N):
      if not i in lines: # i cannot be in list of indices used for test
        trainF.write(corpus[i])
  else: # we just select a single line from the corpus for our test corpus:
    line = args.line - 1 # account for 0-indexing
    if line > N:
      sys.stderr.write("Line out of bounds! (%d lines in corpus)\n"%N)
      sys.exit(1)

    testF.write(corpus[line]) # write this line to the test corpus

    # write all other lines to train corpus:
    for i in xrange(0,N):
      if i != line:
        trainF.write(corpus[i])

  f.close()
  trainF.close()
  testF.close()

