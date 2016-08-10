# -*- coding: utf-8 -*-

from collections import defaultdict
import numpy as np
import sys

from . import _common as common
from . import STOP

cimport cython
cimport numpy as np

#np.set_printoptions(threshold=np.inf)

""" A Hidden Markov Model constructed from hidden (unlabeled) data """
cdef class HiddenDataHMM:
  cdef public _sentences, _states, _labelHash, _sigma, _tau, _smooth
  cdef int _ITER_CAP, _numStates, _wc
  cdef public int _STOPTAG
  cdef float _alpha

  """ Construct the HMM object using a list of outputs and a set of posLabels. """
  def __init__(self, outputs, posLabels, wordCount, labelHash=None):
    # hash to create an array of ints
    self._sentences = [[hash(x) for x in sentence] for sentence in outputs]
    self._numStates = len(posLabels)
    self._states = range(0, self._numStates) # faster np.array indexing
    self._wc = wordCount

    # labelHash maps the string label name to an internal int index
    self._labelHash = labelHash or common.makeLabelHash(posLabels)

    self._STOPTAG = self._labelHash[STOP] # which one is the stop tag?

    # initialise sigmas as random matrix
    randMat = np.random.uniform(0.9,1.1,[self._numStates]*2)
    self._sigma = np.full([self._numStates]*2, 0.1)*randMat # [y,y']->proba

    # initialise tau as defaultdict with random initialisation (default)
    self._alpha = 1.0
    self._smooth = lambda: self._alpha/self._wc
    self._tau = defaultdict(self._smooth)

  """ Compute alphas for this timestep.
        alpha: n x m slice of the alphaBetaMat (n words by m states)
        x: the output at this timestep
      Return: Nothing, calculated in place
  """
  @cython.boundscheck(False)
  cdef void _computeAlphasTimestep(self, double[:,:] sigma, double[:,:] alpha, int i, long x):
    cdef double val
    for y in range(0, self._numStates):
      val = 0.0
      for yprime in range(0, self._numStates):
        val += alpha[(i-1), yprime]*sigma[yprime,y]
      alpha[i,y] = val*self._tau[(y,x)]

  """ Compute alphas for this timestep.
        beta: n x m slice of the alphaBetaMat (n words by m states)
        xNext: the output at next timestep
      Return: Nothing, calculated in place
  """
  @cython.boundscheck(False)
  cdef void _computeBetasTimestep(self, double[:,:] sigma, double[:,:] beta, int i, long xNext):
    cdef double val
    cdef int y, yprime
    for y in range(0, self._numStates):
      val = 0.0
      for yprime in range(0, self._numStates):
        val += beta[(i+1),yprime]*sigma[y,yprime]*self._tau[(yprime,xNext)]
      beta[i,y] = val

  """ Normalise alpha_i and beta_i (both row vectors) """
  @cython.cdivision(True)
  cdef void _normaliseAlphaBeta(self, np.ndarray[double] alpha_i, np.ndarray[double] beta_i):
    normFactor = np.sum(alpha_i)
    alpha_i = alpha_i/normFactor
    beta_i = beta_i/normFactor

  """ Compute E[n_{i,y,x}|x].
        alpha_y: alpha_y(i)
        beta_y: beta_y(i)
        totalProb: alpha_STOP(n)
  """
  @cython.cdivision(True)
  cdef double _expEmissionFreq(self, double alpha_y, double beta_y, double totalProb):
    return alpha_y*beta_y/totalProb

  """ Compute E[N_{i,y,y'}|x].
        alpha_y: alpha_y(i)
        beta_yprime: beta_{y'}(i+1)
        sigma: sigma_{y,y'}
        tau: tau_{y',x_{i+1}}
        totalProb: alpha_STOP(n)
  """
  @cython.cdivision(True)
  cdef double _expTransitionFreq(self, double alpha_y, double beta_yprime,
                                 double sigma, double tau, double totalProb):
    return alpha_y*sigma*tau*beta_yprime/totalProb

  """ Verify model consistency in computing forward/backward probabilities. """
  cdef int _verifyProbs(self, double[:] alpha, double[:] beta, double totalProb):
    cdef double alpha_y, beta_y, prob
    cdef int y
    prob = 0.0
    for y in range(0, self._numStates):
      prob += self._expEmissionFreq(alpha[y], beta[y], totalProb)

    if abs(1.0 - prob) > 1e-6: # probabilities should sum to 1
      return 0
    return 1

  """ Perform the E-Step of EM. Return the expectations. """
  cdef void _do_EStep(self, object expected_yx, double[:,:] expected_yy_, double[:] expected_ycirc,
                       int iteration, int iter_cap):
    # s: sentence, n_sentence: # sentences, n: len(sentence), ALPHA=0, BETA=1, j:beta index
    cdef int s, n_sentence, n, ALPHA, BETA, i, j
    cdef long x_i, x_i1, x_j1
    cdef int STOPTAGIDX = self._STOPTAG
    cdef double alpha_y, beta_y, totalProb, sigma, tau, expOutputFreq

    cdef np.ndarray[double, ndim=2] alphas, betas
    cdef np.ndarray[double, ndim=3] alphaBetaMat

    cdef double[:,:] sigmaMat = self._sigma # memoryview on numpy array
    ALPHA, BETA = 0, 1 # indices

    s = 1
    n_sentence = len(self._sentences) 
    for sentence in self._sentences:
      print "- sentence: %i of %i \t\t (iteration %i/%i)"%(s,n_sentence,iteration,iter_cap)
      sys.stdout.flush()
      s+=1

      n = len(sentence)

      alphaBetaMat = np.zeros([2, n, self._numStates]) # [alpha or beta][timestep][state] -> prob.
      alphaBetaMat[ALPHA,0,STOPTAGIDX] = 1.0
      alphaBetaMat[BETA,(n-1),STOPTAGIDX] = 1.0

      # iterate over sentence without initial STOP for alpha, last STOP for beta
      # e.g. [STOP, "hello", "world", STOP]
      # Calculate alpha and beta using our sigmas and taus
      alphas = alphaBetaMat[ALPHA,:,:]
      betas = alphaBetaMat[BETA,:,:]

      for i in xrange(1,n): # compute fwd/backward probs
        j = n - i - 1
        x_i = sentence[i]
        x_j1 = sentence[j+1]
        self._computeAlphasTimestep(sigmaMat, alphas, i, x_i) # compute alphas for this timestep
        self._computeBetasTimestep(sigmaMat, betas, j, x_j1) # compute betas for this timestep

      for i in xrange(0,n): # normalise fwd/backward probs
        self._normaliseAlphaBeta(alphas[i], betas[i])
        #if self._verifyProbs(alphas[i], betas[i], betas[0][STOPTAGIDX]) == 0:
        #  sys.stderr.write("Probabilities not valid!\n")

      if alphas[(n-1),STOPTAGIDX] == 0.0: # problem
        print alphas
        sys.exit(1)

      # Here we go again, now to calculate expectations
      for i in xrange(n-1): # for each word in the sentence
        x_i = sentence[i]
        x_i1 = sentence[i+1]
        for y in range(0,self._numStates): # iterate over each label y
          alpha_y = alphas[i,y]
          beta_y = betas[i,y]
          totalProb = alphas[(n-1),STOPTAGIDX]

          expOutputFreq = self._expEmissionFreq(alpha_y, beta_y, totalProb)
          expected_yx[(y,x_i)] += expOutputFreq
          expected_ycirc[y] += expOutputFreq

          for y_ in range(0,self._numStates): # iterate over y' for E[n_{y,y'}|x]
            beta_y_ = betas[(i+1),y_]
            sigma = sigmaMat[y,y_]
            tau = self._tau[(y_,x_i1)]
            expected_yy_[y,y_] += self._expTransitionFreq(alpha_y,beta_y_,sigma,tau,totalProb)

  """ Perform the M-Step of EM. Update sigma and tau mappings using expectations. """ 
  cdef void _do_MStep(self, expected_yx, double[:,:] expected_yy_, double[:] expected_ycirc):
    cdef double[:,:] sigmaMat = self._sigma # memoryview on numpy array
    cdef int y, yprime
    cdef float alpha = self._alpha
    for y in range(0,self._numStates):
      for yprime in range(0,self._numStates):
        # smooth sigma:
        sigmaMat[y,yprime] = (expected_yy_[y,yprime]+alpha)/(expected_ycirc[y]+alpha*self._numStates)

    for emission,expectation in expected_yx.iteritems():
      y, _ = emission
      # smooth tau:
      self._tau[emission] = (expectation+alpha)/(expected_ycirc[y] + alpha*self._numStates)
      #self._tau[emission] = (expectation)/(expected_ycirc[y])

  cdef void _train(self, int ITER_CAP, tuple visible_params):
    cdef int i = 1
    cdef double[:,:] e_yy_ # E[n_{y,y'}|x]: (y,y')->float
    cdef double[:] e_ycirc # E[n_{y,\circ}|x]: y->float

    print "Beginning train iterations (EM)..."
    start_expectations = defaultdict(float), np.zeros([self._numStates]*2), np.zeros(self._numStates)
    if visible_params:
      start_distribution, start_expectations = visible_params
      self._sigma, self._tau = start_distribution
      exp_yx, exp_yy_, exp_ycirc = start_expectations # temporarily unpack these to weight them
      # weight these expectation counts because they are more correct:
      exp_yy_ = exp_yy_*100.0
      exp_ycirc = exp_ycirc*100.0
      for key,val in exp_yx.iteritems():
        exp_yx[key] = val*100.0

      self._tau._wc = self._wc # update taudict's internal wordcount
      self._alpha = self._tau._alpha

    while i <= ITER_CAP:
      print "iteration %i" % i

      e_yx, e_yy_, e_ycirc = start_expectations

      # (E-step):
      self._do_EStep(e_yx, e_yy_, e_ycirc, i, ITER_CAP)

      # (M-step): update sigma and tau
      self._do_MStep(e_yx, e_yy_, e_ycirc)

      i += 1 # increment iterations count

  """ Train the HMM using EM to estimate sigma and tau distributions.
        params: (visible_distribution, visible_counts) where
                visible_distribution=(sigma,tau) of the VisibleDataHMM
                visible_counts=(n_yy_, n_ycirc) of the VisibleDataHMM

                pass None if these are not available (for purely unsupervised)
  """
  def train(self, params):
    iter_cap, visible_params = params
    self._train(iter_cap, visible_params)
    print "Done."

  """ Return sigma_{y,yprime} - transition prob. from state y->yprime """
  def getSigma(self, y, yprime):
    y = self._labelHash[y]
    yprime = self._labelHash[yprime]
    return self._sigma[y,yprime]

  """ Compute tau_{y,x} - emission prob. of state y->output x """
  def getTau(self, y, x):
    y = self._labelHash[y]
    x = hash(x)
    return self._tau[(y,x)]

  """ Return a copy of the model's labels """
  def getLabels(self):
    return set(self._labelHash.keys())

  """ Return a copy of the internal mapping of str y -> int i """
  def getLabelHash(self):
    return dict(self._labelHash)

  """ Return a copy of the trained internal distributions sigma and tau """
  def getDistribution(self):
    return (np.copy(self._sigma), dict(self._tau))

