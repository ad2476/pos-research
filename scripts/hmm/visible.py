# -*- coding: utf-8 -*-

from collections import defaultdict
import itertools
import numpy as np
import sys

from . import _common as common

""" A Hidden Markov Model constructed from visible data """
class VisibleDataHMM:

  """ Construct the HMM object using the outputs and labels (and wordcounts)
      Pass a dict of word->count for *UNK* substitution in train()
  """
  def __init__(self, outputs, labels, counts):
    # hash x for compatibility with HiddenDataHMM:
    self._outputs = [[hash(x) for x in sentence] for sentence in outputs]
    self._labels = labels # list of list of states
    self._counts = counts

    self._sigma = None # not yet defined - don't know how many states there are
    self._tau = defaultdict(float)
    self.xSet = set() # set of unique outputs
    self.n_sentences = len(labels)
    if self.n_sentences != len(outputs): # problem
      raise ValueError("Outputs and labels should be the same size")

    self.unkCount = 0 # used for metrics calculation (e.g. % UNK in doc)

    self._alpha = 1.0 # add-alpha

  """ Train the HMM by building the sigma and tau mappings.
        - params is a useless parameter, for conforming to the interface for HMMs
  """
  def train(self, params=None):
    n_yy_ = defaultdict(int) # n_y,y' (number of times y' follows y)
    n_ycirc = defaultdict(int) # n_y,o (number of times any label follows y)
    self.n_yx = defaultdict(int) # n_y,x (number of times label y labels output x)

    # build counts
    unkHash = hash("*UNK*")
    for words,tags in itertools.izip(self._outputs,self._labels):
      n = len(words)
      for i in xrange(0, n - 1):
        y = tags[i]
        y_ = tags[i+1] # y_ = y'
        ytuple = (y,y_)

        x = words[i] # corresponding output
        if self._counts[x] == 1:
          yunk = (y,unkHash)
          self.n_yx[yunk] += 1
          self.xSet.add(unkHash)
          
        yxtuple = (y,x)

        n_yy_[ytuple] += 1
        n_ycirc[y] += 1

        self.n_yx[yxtuple] += 1

        self.xSet.add(x)

    self.tagset = n_ycirc.keys()
    self._labelHash = common.makeLabelHash(self.tagset)
    self.tagsetSize = len(self.tagset)

    # compute sigma matrix:
    self._sigma = np.zeros([self.tagsetSize]*2)
    for y,_ in n_ycirc.iteritems(): # first, set up smoothing:
      yhash = self._labelHash[y]
      sigmaSmoothUnk = self._alpha/(n_ycirc[y]+self._alpha*self.tagsetSize)
      self._sigma[yhash,:] = sigmaSmoothUnk # smooth sigma if the pair (y,y') dne
    for pair,count in n_yy_.iteritems(): # next, initialise sigmas for known (y,y') pairs
      y,yprime = pair
      yhash, yprimehash = self._labelHash[y],self._labelHash[yprime]
      self._sigma[yhash,yprimehash] = (count+self._alpha)/(n_ycirc[y]+self._alpha*self.tagsetSize)
    
    # compute tau dict:
    for pair,count in self.n_yx.iteritems():
      y,x = pair
      yhash = self._labelHash[y]
      self._tau[(yhash,x)] = count/float(n_ycirc[y])

  """ Return the sigma_{y,y'} for given y and y' - or 0 if dne """
  def getSigma(self, y, yprime):
    yhash = self._labelHash[y]
    yprimehash = self._labelHash[yprime]
    return self._sigma[yhash,yprimehash]

  """ Return the tau_{y,x} for given y and x - or 0 if dne """
  def getTau(self, y, x):
    y = self._labelHash[y]
    x = hash(x)
    if x not in self.xSet:
      x = hash("*UNK*") # treat unknowns as "*UNK*" when decoding

    return self._tau[(y,x)]

  """ Return a copy of the labels of this HMM """
  def getLabels(self):
    return set(self.tagset)

  """ Return a copy of the internal mapping of str y -> int i """
  def getLabelHash(self):
    return dict(self._labelHash)

  """ Return the trained internal distributions sigma and tau """
  def getDistribution(self):
    return (self._sigma, self._tau)

