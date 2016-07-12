from collections import defaultdict
import random
import itertools

STOP = "**@sToP@**" # The stop symbol

""" A Hidden Markov Model constructed from hidden (unlabeled) data """
class HiddenDataHMM:

  """ Construct the HMM object using a list of outputs and a set of posLabels. """
  def __init__(self, outputs, posLabels):
    self._sentences = outputs # list of list of words, each elem is a sentence (as list)
    self._states = posLabels
    self._ITER_CAP = 10

    self._sigma = defaultdict(self._paramSmooth)
    self._tau = defaultdict(self._paramSmooth)

    self.unkCount = 0 # used for metrics calculation (e.g. % UNK in doc)

    random.seed()

  """ Custom smoothing function for the defaultdicts _sigma and _tau """
  def _paramSmooth(pair):
    return 0.01*random.uniform(0.95,1.05)

  """ Compute alpha for the current timestep
        prevAlpha: dictionary of alpha(i-1): y->float
        x: the output at this timestep
      Return: float (to be added to ith iteration's alpha(i) dict)
  """
  def _computeAlpha(self, prevAlpha, x):
    alpha = defaultdict(float)
    if prevAlpha is None: # base case
      alpha[STOP] = 1.0
    else:
      for y in self._states:
        val = 0
        for yprime in self._states:
          val += prevAlpha[yprime]*self._sigma[(yprime,y)]
        alpha[y] = val*self._tau[(y,x)]

    return alpha

  """ Compute beta for the current timestep
        prevBeta: dictionary of beta(i+1): y->float
        xNext: x_{i+1}, i.e. the output one position ahead of this timestep
      Return: float
  """
  def _computeBeta(self, prevBeta, xNext):
    beta = defaultdict(float)
    if prevBeta is None:
      beta[STOP] = 1.0
    else:
      for y in self._states:
        val = 0
        for yprime in self._states:
          val += prevBeta[yprime]*self._sigma[(y,yprime)]*self._tau[(yprime,xNext)]
        beta[y] = val

    return beta

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

  """ Train the HMM using EM to estimate sigma and tau distributions.
        start_distribution: tuple (sigma, tau) defaultdicts representing an
                            initial distribution (optional)
  """
  def train(self, start_distribution=None):
    print "Beginning train iterations (EM)..."
    if start_distribution:
      self._sigma, self._tau = start_distribution

    iterations = 0
    while iterations < self._ITER_CAP:
      print "iteration %i" % iterations
      expected_yy_ = defaultdict(float) # E[n_{y,y'}|x]: (y,y')->float
      expected_yx = defaultdict(float) # E[n_{y,x}|x]: (y,x)->float
      expected_ycirc = defaultdict(float) # E[n_{y,\circ}|x]: y->float

      # (E-step):
      s = 0
      for sentence in self._sentences:
        print "- sentence: %i" % s
        s+=1

        n = len(sentence)
        alpha = [None for _ in range(0,n)]
        beta = [None for _ in range(0,n)]

        alpha[0] = self._computeAlpha(None, STOP) # starts from the front
        beta[n-1] = self._computeBeta(None, None) # starts from the back

        # iterate over sentence without initial STOP for alpha, last STOP for beta
        # e.g. [STOP, "hello", "world", STOP]
        # Calculate alpha and beta using our sigmas and taus
        print "-- compute alpha and beta:"
        for i in xrange(1,n):
          print "--- word %i" % i
          j = n - i - 1
          x_i = sentence[i]
          x_j1 = sentence[j+1]
          alpha[i] = self._computeAlpha(alpha[i-1], x_i)
          beta[j] = self._computeBeta(beta[j+1], x_j1)

        # Here we go again, now to calculate expectations
        print "-- compute expectations:"
        for i in xrange(0,n-1):
          print "--- word %i" %i
          x = sentence[i]
          nextX = sentence[i+1]
          for y in self._states:
            alpha_y = alpha[i][y]
            beta_y = beta[i][y]
            totalProb = alpha[n-1][STOP]
            expOutputFreq = self._expEmissionFreq(alpha_y, beta_y, totalProb)
            expected_yx[(y,x)] += expOutputFreq
            expected_ycirc[y] += expOutputFreq # TODO: idk if this is right??

            for y_ in self._states:
              beta_y_ = beta[i+1][y_]
              sigma = self._sigma[(y,y_)]
              tau = self._tau[(y_,nextX)]
              expected_yy_[(y,y_)] += self._expTransitionFreq(alpha_y,beta_y_,sigma,tau,totalProb)

      # (M-step): update sigma and tau
      for transition,expectation in expected_yy_.iteritems():
        y, _ = transition
        self._sigma[transition] = expectation/expected_ycirc[y]

      for emission,expectation in expected_yx.iteritems():
        y, _ = emission
        self._tau[emission] = expectation/expected_ycirc[y]

      iterations += 1 # increment iterations count

  def getSigma(self, y, yprime):
    return self._sigma[(y,yprime)]

  def getTau(self, y, x):
    return self._tau[(y,x)]

  def getLabels(self):
    return self._states

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
    return list(self.n_ycirc.keys())

