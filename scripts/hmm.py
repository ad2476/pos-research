from collections import defaultdict
import random
import itertools
import numpy as np

STOP = "**@sToP@**" # The stop symbol

""" A Hidden Markov Model constructed from hidden (unlabeled) data """
class HiddenDataHMM:

  """ Construct the HMM object using a list of outputs and a set of posLabels. """
  def __init__(self, outputs, posLabels, labelHash=None):
    self._ITER_CAP = 1

    #self._sentences = [[hash(x) for x in sentence] for sentence in outputs]
    self._sentences = outputs
    self._numStates = len(posLabels)
    self._states = xrange(0,self._numStates) # faster np.array indexing

    if not labelHash:
      self._labelHash = {} # map actual label to its internal index number
      for each in enumerate(posLabels):
        i,y = each
        self._labelHash[y] = i
    else:
      self._labelHash = labelHash

    self._STOPTAG = self._labelHash[STOP] # which one is the stop tag?

    self._sigma = np.full([self._numStates]*2, 0.01)*np.random.uniform(0.95,1.05) # [y,y']->prob
    self._tau = defaultdict(self._paramSmooth)

    self.unkCount = 0 # used for metrics calculation (e.g. % UNK in doc)

    random.seed()

  """ Custom smoothing function for the defaultdicts _sigma and _tau """
  def _paramSmooth(pair):
    return 0.01*random.uniform(0.95,1.05)

  """ Compute alphas for this timestep.
        alpha: n x m slice of the alphaBetaMat (n words by m states)
        x: the output at this timestep
      Return: Nothing, calculated in place
  """
  def _computeAlphasTimestep(self, alpha, i, x):
    for y in self._states:
      val = 0.0
      for yprime in self._states:
        val += alpha[i-1][yprime]*self._sigma[yprime,y]
      alpha[i][y] = val*self._tau[(y,x)]

  """ Compute alphas for this timestep.
        beta: n x m slice of the alphaBetaMat (n words by m states)
        xNext: the output at next timestep
      Return: Nothing, calculated in place
  """
  def _computeBetasTimestep(self, beta, i, xNext):
    for y in self._states:
      val = 0.0
      for yprime in self._states:
        val += beta[i+1][yprime]*self._sigma[y,yprime]*self._tau[(yprime,xNext)]
      beta[i][y] = val

  """ Normalise alpha_i and beta_i (both row vectors) """
  def _normaliseAlphaBeta(self, alpha_i, beta_i):
    normFactor = np.sum(alpha_i)
    alpha_i = alpha_i/normFactor
    beta_i = beta_i/normFactor

  """ Not used rn, not compat with np structure!! """
  def _forwardBackward(self, prevFwdBwd, x_i, x_j1):
    prevAlpha, prevBeta = prevFwdBwd
    alpha, beta = np.zeros(self._numStates), np.zeros(self._numStates)

    for y in self._states:
      alpha_val = 0
      beta_val = 0
      for yprime in self._states:
        alpha_val += prevAlpha[yprime]*self._sigma[(yprime,y)]
        beta_val += prevBeta[yprime]*self._sigma[(y,yprime)]*self._tau[(yprime,x_j1)]
      alpha[y] = alpha_val*self._tau[(y,x_i)]
      beta[y] = beta_val

    return (alpha, beta)

  """ Compute E[n_{i,y,x}|x].
        alpha_y: alpha_y(i)
        beta_y: beta_y(i)
        totalProb: alpha_STOP(n)
  """
  def _expEmissionFreq(self, alpha_y, beta_y, totalProb):
    return float(alpha_y*beta_y)/totalProb

  """ Compute E[N_{i,y,y'}|x].
        alpha_y: alpha_y(i)
        beta_yprime: beta_{y'}(i+1)
        sigma: sigma_{y,y'}
        tau: tau_{y',x_{i+1}}
        totalProb: alpha_STOP(n)
  """
  def _expTransitionFreq(self, alpha_y, beta_yprime, sigma, tau, totalProb):
    return float(alpha_y*sigma*tau*beta_yprime)/totalProb

  """ Perform the E-Step of EM. Return the expectations. """
  def _do_EStep(self, iteration):
    expected_yy_ = np.zeros([self._numStates]*2) # E[n_{y,y'}|x]: (y,y')->float
    expected_yx = defaultdict(float) # E[n_{y,x}|x]: (y,x)->float
    expected_ycirc = np.zeros(self._numStates) # E[n_{y,\circ}|x]: y->float

    s = 1
    n_sentence = len(self._sentences)
    for sentence in self._sentences:
      print "- sentence: %i of %i \t\t (iteration %i/%i)" % (s, n_sentence, iteration, self._ITER_CAP)
      s+=1

      n = len(sentence)

      ALPHA, BETA = 0, 1 # indices
      alphaBetaMat = np.zeros([2, n, self._numStates]) # [alpha or beta][timestep][state] -> prob.

      alphaBetaMat[ALPHA][0][self._STOPTAG] = 1.0
      alphaBetaMat[BETA][n-1][self._STOPTAG] = 1.0

      # iterate over sentence without initial STOP for alpha, last STOP for beta
      # e.g. [STOP, "hello", "world", STOP]
      # Calculate alpha and beta using our sigmas and taus
      print "-- compute alpha and beta:"
      alphas = alphaBetaMat[ALPHA,:,:]
      betas = alphaBetaMat[BETA,:,:]

      for i in xrange(1,n):
        print "--- word %i" % i
        j = n - i - 1
        x_i = sentence[i]
        x_j1 = sentence[j+1]
        self._computeAlphasTimestep(alphas, i, x_i) # compute alphas for this timestep
        self._computeBetasTimestep(betas, j, x_j1) # compute betas for this timestep

      for i in xrange(1,n):
        self._normaliseAlphaBeta(alphas[i], betas[i])

      # Here we go again, now to calculate expectations
      print "-- compute expectations:"
      for i in xrange(0,n-1):
        print "--- word %i" %i
        x = sentence[i]
        nextX = sentence[i+1]
        for y in self._states:
          alpha_y = alphas[i][y]
          beta_y = betas[i][y]
          totalProb = alphas[n-1][self._STOPTAG]
          expOutputFreq = self._expEmissionFreq(alpha_y, beta_y, totalProb)
          expected_yx[(y,x)] += expOutputFreq
          expected_ycirc[y] += expOutputFreq

          for y_ in self._states: # iterate over y' for E[n_{y,y'}|x]
            beta_y_ = betas[i+1][y_]
            sigma = self._sigma[y,y_]
            tau = self._tau[(y_,nextX)]
            expected_yy_[y,y_] += self._expTransitionFreq(alpha_y,beta_y_,sigma,tau,totalProb)

    return (expected_yx, expected_yy_, expected_ycirc) # return expectations

  """ Perform the M-Step of EM. Update sigma and tau mappings using expectations. """ 
  def _do_MStep(self, expected_yx, expected_yy_, expected_ycirc):
    for y in self._states:
      for yprime in self._states:
        self._sigma[y,yprime] = expected_yy_[y,yprime]/expected_ycirc[y]

    for emission,expectation in expected_yx.iteritems():
      y, _ = emission
      self._tau[emission] = expectation/expected_ycirc[y]

  """ Train the HMM using EM to estimate sigma and tau distributions.
        start_distribution: tuple (sigma, tau) defaultdicts representing an
                            initial distribution (optional)
  """
  def train(self, start_distribution=None):
    print "Beginning train iterations (EM)..."
    if start_distribution:
      self._sigma, self._tau = start_distribution

    i = 1
    while i <= self._ITER_CAP:
      print "iteration %i" % i

      # (E-step):
      e_yx, e_yy_, e_ycirc = self._do_EStep(i)

      # (M-step): update sigma and tau
      self._do_MStep(e_yx, e_yy_, e_ycirc)

      i += 1 # increment iterations count

    print self._sigma

  def getSigma(self, y, yprime):
    y = self._labelHash[y]
    yprime = self._labelHash[yprime]
    return self._sigma[y,yprime]

  def getTau(self, y, x):
    y = self._labelHash[y]
    return self._tau[(y,x)]

  def getState(self, tag): # Return the internal state value corresponding to the given POS tag
    return self._labelHash[tag]

  def getLabels(self):
    return self._labelHash.keys()

""" A Hidden Markov Model constructed from visible data """
class VisibleDataHMM:

  """ Construct the HMM object using the outputs and labels
      better: True for better_tag, False for tag
  """
  def __init__(self, outputs, labels, better):
    self._outputs = outputs # list of list of words
    self._labels = labels # list of list of states
    self._better = better

    self.sigma = {}
    self.tau = {}
    self.xSet = {} # set of unique outputs
    self.n_sentences = len(labels)
    if self.n_sentences != len(outputs): # problem
      raise ValueError("Outputs and labels should be the same size")

    self.unkCount = 0 # used for metrics calculation (e.g. % UNK in doc)

    self._trained = False

    self._alpha = 1 # add-alpha

  """ Train the HMM by building the sigma and tau mappings.
      Pass this method a dict of word->count for *UNK* substitution
  """
  def train(self, counts):
    self.n_yy_ = {} # n_y,y' (number of times y' follows y)
    self.n_ycirc = {} # n_y,o (number of times any label follows y)
    self.n_yx = {} # n_y,x (number of times label y labels output x)

    # build counts
    for words,tags in itertools.izip(self._outputs,self._labels):
      n = len(words)
      for i in xrange(0, n - 1):
        y = tags[i]
        y_ = tags[i+1] # y_ = y'
        ytuple = (y,y_)

        x = words[i] # corresponding output
        if counts[x] == 1 and self._better:
          yunk = (y,"*UNK*")
          self.n_yx[yunk] = self.n_yx.get(yunk, 0) + 1
          self.xSet["*UNK*"] = True
          
        yxtuple = (y,x)

        self.n_yy_[ytuple] = self.n_yy_.get(ytuple, 0) + 1
        self.n_ycirc[y] = self.n_ycirc.get(y, 0) + 1

        self.n_yx[yxtuple] = self.n_yx.get(yxtuple, 0) + 1

        self.xSet[x] = True

    # build sigmas
    self.tagsetSize = len(self.n_ycirc.keys())
    
  """ Return the sigma_{y,y'} for given y and y' - or 0 if dne """
  def getSigma(self, y, yprime):
    ytuple = (y,yprime)
    return (self.n_yy_.get(ytuple, 0)+self._alpha)/float(self.n_ycirc.get(y,0) + self._alpha*self.tagsetSize)


  """ Return the tau_{y,x} for given y and x - or 0 if dne """
  def getTau(self, y, x):
    count = 0
    if x in self.xSet and x is not "*UNK*":
      count = self.n_yx.get((y,x), 0)
    elif self._better: # x is *UNK* and we're using better_tag
      count = self.n_yx.get((y,"*UNK*"), 0)
    else: # x is *UNK* and we're not using better_tag
      count = 1 # tau_{y,*U*} = 1

    tau = count/float(self.n_ycirc[y])
    return tau

  """ Return a copy of the labels of this HMM """
  def getLabels(self):
    return set(self.n_ycirc.keys())

