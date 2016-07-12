import math

class UnigramLangmod:

  """ Constructor sets some admin stuff """
  def __init__(self, train, heldout):
    self.STOP = "**@sToP@**" # The stop symbol
    self.trainCorpus = train
    self.heldoutCorpus = heldout
    self._trained = False

  """ Build unigram counts from a corpus, aka n_w(d)
      Inputs:
        - document: An input corpus to build counts for
                    This can be a list of string, or a file object
      Returns:
        - A tuple of (counts, totalWordCount)
  """
  def buildCounts(self, document):
    counts = {} # map string -> int: aka n_w(d)

    # Populate n_w(d) aka counts dict
    n_o = 0
    for line in document:
      words = line
      for word in words:
        n = counts.get(word, 0) + 1
        counts[word] = n
        n_o += 1

    if isinstance(document, file):
      document.seek(0) # Return file pointer to beginning

    return (counts, n_o)

  """ Train a smoothed unigram model given n_w and the set of words, W.
       This builds the mapping \\theta_w of smoothed log probabilities
      Inputs:
        - alpha (optional): Specify an alpha value, or none to use optimal alpha
      Returns: Nothing
  """
  def train(self, alpha=None):
    self.n_w, self.wordCount = self.buildCounts(self.trainCorpus)
    self.uniqueWords = len(self.n_w) + 1 # include *U* - this is |W|

    if alpha is None:
      self.alpha = self._optimSmoothParams()
    else:
      self.alpha = alpha

    self._trained = True

  """ Compute the log likelihood of a given document using this model
      Inputs:
        - document: Input document
      Returns: Log likelihood of given document, or None if model isn't trained
  """
  def logLikelihood(self, document):
    logProb = None

    if self._trained:
      # First, get the count of words in the input corpus
      docCounts, _ = self.buildCounts(document)

      # Next, we iterate over the words in the input corpus
      # to compute log likelihood
      logProb = -1*self._likelihoodFunction(docCounts, self.alpha)

    return logProb

  def getSmoothParams(self):
    return self.alpha

  def textSmoothParams(self):
    retstr = "alpha: " + str(self.alpha)
    return retstr

  """ Find the best alpha that maximises the likelihood of the held-out
       data using golden-section search
      Inputs: None
      Returns: The best alpha i.e. argmax_a L_d(theta)
  """
  def _optimSmoothParams(self):
    heldoutCounts, _ = self.buildCounts(self.heldoutCorpus)

    goldenRatio = (math.sqrt(5) - 1)/2

    start, end = (1e-10, 1e10)
    c = end - goldenRatio*(end - start) 
    d = start + goldenRatio*(end - start)

    while abs(c - d) > 1e-6:
      right = self._likelihoodFunction(heldoutCounts, c)
      left = self._likelihoodFunction(heldoutCounts, d)
      if right > left:
        end = d
        d = c
        c = end - goldenRatio*(end - start)
      else:
        start = c
        c = d
        d = start + goldenRatio*(end - start)

    return (start+end)/2

  """ The function tilde{theta}_{w}
      Inputs:
        - w: A word w to find the probability of
        - alpha: The smoothing factor
      Returns:
        - The smoothed probability of w
  """
  def _thetaFunction(self, w, alpha):
    n_w = self.n_w.get(w, 0)
    return math.log(n_w + alpha) - math.log(self.wordCount + alpha*self.uniqueWords) 

  """ Computes the positive log prob likelihood given docCounts and alpha """
  def _likelihoodFunction(self, docCounts, alpha):
    logProb = 0.0
    for w,count in docCounts.iteritems():
      logProb += count*self._thetaFunction(w, alpha)

    return logProb
