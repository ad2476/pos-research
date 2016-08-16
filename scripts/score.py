import sys
import argparse
import itertools
import preparser

def parseProgramArgs():
  parser = argparse.ArgumentParser(description="Score tagged output.")
  parser.add_argument("gold", help="Gold (correct) output.")
  parser.add_argument("tagged", help="Tagged output.")
  parser.add_argument("--lang", choices=["EN", "SANS"], required=True,
                      help="Select tagging language. Required.")

  return parser.parse_args()

def calculateAccuracy(filePreparser, correctFile, estimateFile):
  numCorrect = 0.0
  total = 0.0
  for correct,estimate in itertools.izip(correctFile, estimateFile):
    correctTags = filePreparser.getSentenceTags(correct)
    estimateTags = filePreparser.getSentenceTags(estimate)
    total += len(estimateTags)
    if len(correctTags) == len(estimateTags):
      for i in xrange(0, len(correctTags)):
        if correctTags[i] == estimateTags[i]:
          numCorrect += 1.0

  if total == 0.0:
    return 0.0

  return numCorrect/total

if __name__ == '__main__':

  args = parseProgramArgs()

  testFile = open(args.gold, 'r')
  outputFile = open(args.tagged, 'r')

  if args.lang == "EN":
    FilePreparser = preparser.EnglishWSJParser
  elif args.lang == "SANS":
    FilePreparser = preparser.SanskritJNUParser

  accuracy = calculateAccuracy(FilePreparser, testFile, outputFile)
  print accuracy

  testFile.close()
  outputFile.close()

