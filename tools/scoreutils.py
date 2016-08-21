import itertools
import numpy as np
from collections import defaultdict

"""Map the association between a pos label and its integer index.
   Necessary because hmms store labels as ints for faster indexing on numpy arrays,
    but there needs to be a way to know what label maps to what internal index.
"""
def makeLabelHash(labels):
 labelHash = {}
 for i,y in enumerate(labels):
  labelHash[y] = i

 return labelHash

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

""" Computes the confusion matrix for the tagged output """
def calculateConfusion(filePreparser, gold, tagged, tagset=None):
  confusionCounts = defaultdict(int) # (ygold,ypred)->count
  labels = set()

  for correct,estimate in itertools.izip(gold, tagged):
    correctTags = filePreparser.getSentenceTags(correct)
    estimateTags = filePreparser.getSentenceTags(estimate)

    for i,y in enumerate(correctTags):
      actual = y
      predicted = estimateTags[i]
      confusionCounts[(actual,predicted)] += 1.0 # actual is labeled as predicted
      labels.add(actual)
      labels.add(predicted)

  if tagset:
    labels = list(tagset)
  else:
    labels = list(labels)

  labelHash = makeLabelHash(labels)
  n = len(labels)
  confusionMat = np.zeros([n]*2) # nxn matrix of confusion
  for dazed,confused in confusionCounts.iteritems():
    actual,predicted = dazed
    ygold,ypred = labelHash[actual], labelHash[predicted]
    confusionMat[ygold,ypred] = confused

  # ignore divide by 0 and make it 0
  with np.errstate(divide='ignore', invalid='ignore'):
    norm = np.sum(confusionMat, axis=1).reshape(n,1)
    confusionMat = confusionMat/norm

  return labels, np.nan_to_num(confusionMat)

