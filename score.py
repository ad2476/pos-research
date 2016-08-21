#!/usr/bin/env python2

import sys
import argparse
import itertools
import numpy as np
from collections import defaultdict

from pos import preparser
from pos.hmm import _common

np.set_printoptions(threshold=np.inf,linewidth=77)

def parseProgramArgs():
  parser = argparse.ArgumentParser(description="Score tagged output.")
  parser.add_argument("gold", help="Gold (correct) output.")
  parser.add_argument("tagged", help="Tagged output.")
  parser.add_argument("tagset", help="Path to tagset file")
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

def calculateConfusion(filePreparser, gold, tagged, tagset):
  confusionCounts = defaultdict(int) # (ygold,ypred)->count
  labels = set()

  for correct,estimate in itertools.izip(gold, tagged):
    correctTags = filePreparser.getSentenceTags(correct)
    estimateTags = filePreparser.getSentenceTags(estimate)
    for i,y in enumerate(correctTags):
      actual = y
      predicted = estimateTags[i]
      confusionCounts[(actual,predicted)] += 1.0
      labels.add(actual)
      labels.add(predicted)

  labels = list(labels)
  labelHash = _common.makeLabelHash(labels)
  n = len(labels) # number of label classes
  print n
  confusionMat = np.zeros([n]*2) # nxn matrix of confusion
  for dazed,confused in confusionCounts.iteritems():
    actual,predicted = dazed
    ygold,ypred = labelHash[actual], labelHash[predicted]
    confusionMat[ygold,ypred] = confused

  #norm = np.sum(confusionMat, axis=1).reshape(n,1)

  return labels,confusionMat

if __name__ == '__main__':

  args = parseProgramArgs()

  testFile = open(args.gold, 'r')
  outputFile = open(args.tagged, 'r')
  tagsetFile = open(args.tagset, 'r')

  gold = [line for line in testFile]
  tagged = [line for line in outputFile]
  tagset = [line.split()[0] for line in tagsetFile]

  if args.lang == "EN":
    FilePreparser = preparser.EnglishWSJParser
  elif args.lang == "SANS":
    FilePreparser = preparser.SanskritJNUParser

  labels, confusion = calculateConfusion(FilePreparser, gold, tagged, tagset)
  sys.stdout.write("    ")
  for label in labels:
    sys.stdout.write("%s   "%label)
  sys.stdout.write("\n")

  for label,row in zip(labels,confusion):
    print '%s [%s]' % (label, ' '.join('%03s' % i for i in row))

  accuracy = calculateAccuracy(FilePreparser, gold, tagged)
  print accuracy

  testFile.close()
  outputFile.close()
  tagsetFile.close()

