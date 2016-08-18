import sys
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

  if len(sys.argv) != 3:
    sys.stderr.write("Usage: python corpus.py CORPUS RATIO\n")
    sys.stderr.write("  e.g. `python corpus.py sans/TaggedCorpus.txt 0.2` creates a test and train file, with the test 20% of CORPUS and the train 80% of CORPUS")
    sys.exit(1)

  fname = sys.argv[1] # tagged corpus
  r = float(sys.argv[2]) # ratio

  f = open(fname, "r") # open the corpus for reading
  trainF = open("train.txt", "w")
  testF = open("test.txt", "w")

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

