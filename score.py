import sys
import itertools

def printUsage():
  print "Usage: python score.py <correct> <tagger_output>"

def calculateAccuracy(correctFile, estimateFile):
  
  numCorrect = 0
  total = 0
  for correct,estimate in itertools.izip(correctFile, estimateFile):
    correct = correct.split()
    estimate = estimate.split()
    total += len(estimate)/2.0
    if len(correct) == len(estimate):
      for i in xrange(0, len(correct), 2):
        if correct[i+1] == estimate[i+1]:
          numCorrect += 1

  if total == 0:
    return 0.0

  return numCorrect/float(total)

if __name__ == '__main__':

  if len(sys.argv) != 3:
    printUsage()
    sys.exit(1)

  testFile = open(sys.argv[1], 'r')
  outputFile = open(sys.argv[2], 'r')

  accuracy = calculateAccuracy(testFile, outputFile)

  print accuracy

  testFile.close()
  outputFile.close()

