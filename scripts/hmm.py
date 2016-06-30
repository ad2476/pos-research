# Estimate sigma and tau parameters from data

STOP = "**@sToP@**" # The stop symbol

""" A Hidden Markov Model constructed from visible data """
class VisibleDataHMM:

  """ Construct the HMM object using the outputs and labels
      better: True for better_tag, False for tag
  """
  def __init__(self, outputs, labels, better):
    self._outputs = outputs
    self._labels = labels
    self._better = better

    self.sigma = {}
    self.tau = {}
    self.xSet = {} # set of unique outputs
    self.n = len(labels)
    if self.n != len(outputs): # problem
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
    for i in xrange(0, self.n - 1):
      y = self._labels[i]
      y_ = self._labels[i+1] # y_ = y'
      ytuple = (y,y_)

      x = self._outputs[i] # corresponding output
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
