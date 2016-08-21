#!/usr/bin/env python2

import sys
import argparse

from pos import preparser
from tools.scoreutils import *

np.set_printoptions(threshold=np.inf,linewidth=77)

def parseProgramArgs():
  parser = argparse.ArgumentParser(description="Score tagged output.")
  parser.add_argument("gold", help="Gold (correct) output.")
  parser.add_argument("tagged", help="Tagged output.")
  parser.add_argument("-v", "--verbose", help="Include flag for labels on output values",
                      action="store_true")
  parser.add_argument("--confusion", help="Report confusion matrix",
                      action="store_true")
  parser.add_argument("--accuracy", help="Report accuracy score (# correctly tagged/total # tagged)",
                      action="store_true")
  parser.add_argument("--lang", choices=["EN", "SANS"], required=True,
                      help="Select tagging language. Required.")

  return parser.parse_args()

if __name__ == '__main__':

  args = parseProgramArgs()

  testFile = open(args.gold, 'r')
  outputFile = open(args.tagged, 'r')

  gold = [line for line in testFile]
  tagged = [line for line in outputFile]

  if args.lang == "EN":
    FilePreparser = preparser.EnglishWSJParser
  elif args.lang == "SANS":
    FilePreparser = preparser.SanskritJNUParser

  if args.confusion:
    labels, confusion = calculateConfusion(FilePreparser, gold, tagged)

    #print "Diagonal of confusion matrix:"
    diag = np.diagonal(confusion) # report the diagonal
    for label,val in zip(labels,diag):
      pass
      #print "%s: %.6f" % (label,val)

    if args.verbose:
      sys.stdout.write("Balanced accuracy: ")
    print "%.6f" % (np.sum(diag)/len(diag))

  # report accuracy if flag is explicitly specified, or if --confusion is
  # omitted (or both)
  if not args.confusion or args.accuracy:
    accuracy = calculateAccuracy(FilePreparser, gold, tagged)
    if args.verbose:
      sys.stdout.write("Word-level accuracy: ")
    print accuracy

  testFile.close()
  outputFile.close()

