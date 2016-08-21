
"""Map the association between a pos label and its integer index.
   Necessary because hmms store labels as ints for faster indexing on numpy arrays,
    but there needs to be a way to know what label maps to what internal index.
"""
def makeLabelHash(labels):
 labelHash = {}
 for i,y in enumerate(labels):
  labelHash[y] = i

 return labelHash

""" A subclass of a dict that overrides __missing__ """
class TauDict(dict):

  """ n_ycirc: counts the times a label y maps to any output (n_ycirc :: int -> Number)
      wordCount: an integer/float of the total number of words in the corpus
  """
  def __init__(self, alpha, n_ycirc, wordCount):
    self._alpha = alpha
    self._n_ycirc = n_ycirc
    self._wc = wordCount

  """ Override dict's __missing__ for smoothing taus """
  def __missing__(self, key):
    y,_ = key
    return self._alpha/(self._n_ycirc[y] + self._alpha*self._wc)

