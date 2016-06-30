# Hidden Markov Model decoders

import hmm as HMM

class ViterbiDecoder:

  """ Construct the decoder by passing a hidden markov model. """
  def __init__(self, hmm, wordCounts):
    self.hmm = hmm
    self.wordCounts = wordCounts

  """ Decode a given sentence (as a list)
      
      Returns a sequence of part of speech tags for the input sentence
  """
  def decode(self, sentence):
    decoded = []

    n = len(sentence)

    prev_mu = 1.0
    yprime = HMM.STOP
    states = self.hmm.getLabels()
    for i in xrange(0,n): # iterate over the sentence
      word = sentence[i]
      if (self.wordCounts.get(word, 0) == 0) or (word is "*UNK*"):
        self.hmm.unkCount += 1
        #word = "*UNK*"

      argmax = (None, 0.0)
      for y in states: # argmax over every possible state y
        _, maxmu = argmax
        mu = prev_mu*self.hmm.getSigma(yprime,y)*self.hmm.getTau(y,word)
        #mu = self.hmm.n_yx.get((y,word), 0.0) # "stupid" decoder - most freq.

        if mu >= maxmu:
          argmax = (y,mu)

      # Update step:
      maxy, maxmu = argmax

      prev_mu = maxmu # update prev_mu
      yprime = maxy # update yprime
      decoded.append(maxy) # build decoded sequence

    return decoded

