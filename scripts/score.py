import sys
import itertools
import preparser

def printUsage():
  print "Usage: python score.py <correct> <tagger_output>"

def calculateAccuracy(filePreparser, correctFile, estimateFile):
  
  numCorrect = 0
  total = 0
  for correct,estimate in itertools.izip(correctFile, estimateFile):
    _, correctTags = filePreparser([correct]).parseWordsTags()
    _, estimateTags = filePreparser([estimate]).parseWordsTags()
    correctTags = correctTags[0]
    estimateTags = estimateTags[0]
    total += len(estimateTags)
    if len(correctTags) == len(estimateTags):
      for i in xrange(0, len(correctTags)):
        if correctTags[i] == estimateTags[i]:
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

  #accuracy = calculateAccuracy(preparser.SanskritJNUParser, testFile, outputFile)
  accuracy = calculateAccuracy(preparser.EnglishWSJParser, testFile, outputFile)

  print accuracy

  testFile.close()
  outputFile.close()

