import sys
import itertools
import unigram
import hmm
import decoder
import preparser

better = "--better_tag"

def printUsage():
    print "Tag the words in 'testfile' with their corresponding parts of speech."
    print "Usage: python tagger.py <testfile> <trainfile> <outputfile> [mode]"
    print "\tmode: --tag or --better_tag"

""" Given a sentence list and tag list, format proper output string: """
def taggedSequenceToStr(sentence, tags):
  n = len(sentence)
  output = ""
  for word,tag in itertools.izip(sentence, tags):
    output += word + " " + tag + " "

  return output

if __name__ == '__main__':

  if len(sys.argv) != 5:
    printUsage()
    sys.exit(1)

  trainFile = open(sys.argv[1], 'r')
  testFile = open(sys.argv[2], 'r')
  outFile = open(sys.argv[3], 'w')

  trainData = [line for line in trainFile]
  testData = [line for line in testFile]

  mode = sys.argv[4]

  lm = unigram.UnigramLangmod(None, None) # We don't actually need this for langmod
  counts, _ = lm.buildCounts(trainData) # Build word counts from the input

  WSJPreparser = preparser.EnglishWSJParser(trainData)

  data = None
  try:
    data = WSJPreparser.parseWordsTags()
  except IndexError:
    print "Error parsing input: Bad format"
    sys.exit(1)

  words,tags = data
  try:
    model = hmm.VisibleDataHMM(words, tags, mode == better) # feed data
    model.train(counts) # actually build sigma and tau

    viterbi = decoder.ViterbiDecoder(model, counts)

    # decode the test file:
    for line in testData:
      sentence = line.split()[::2]
      sentence = [word if counts.get(word,0) else "*UNK*" for word in sentence]
      yhat = viterbi.decode(sentence)
      tagged = taggedSequenceToStr(sentence, yhat)
      outFile.write(tagged+"\n")

  except SyntaxError as e:
    print e
    sys.exit(1)

  _, testWordCount = lm.buildCounts(testData)
  print float(model.unkCount)/testWordCount

  trainFile.close()
  testFile.close()
  outFile.close()

