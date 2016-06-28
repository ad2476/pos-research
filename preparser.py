from hmm import STOP

""" This class will pre-parse POS-tagged files in the format of the WSJ data """
class EnglishWSJParser:

  def __init__(self, inputData):
    self._rawdata = inputData

  """ Parse the outputs and tags into separate lists """
  def parseWordsTags(self):
    words = []
    tags = []

    stopPair = STOP + " " + STOP
    for line in self._rawdata:
      line = stopPair + " " + line + stopPair # pad with stop symbols
      sentence = line.split()
      # i increments by 2 from 0 to len(sentence)
      for i in xrange(0, len(sentence), 2):
        word = sentence[i]
        label = sentence[i+1]

        words.append(word)
        tags.append(label)

    return words,tags
