import sys
import unigram
import hmm
import decoder
import preparser

better = "--better_tag"

def printUsage():
    print "Tag the words in 'testfile' with their corresponding parts of speech."
    print "Usage: python tagger.py <testfile> <trainfile> <outputfile> [mode]"
    print "\tmode: --tag or --better_tag"

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

  #FilePreparser = preparser.EnglishWSJParser(trainData)
  FilePreparser = preparser.SanskritJNUParser(trainData)

  data = None
  try:
    data = FilePreparser.parseWordsTags()
  except IndexError:
    print "Error parsing input: Bad format"
    sys.exit(1)

  words,tags = data
  counts, _ = lm.buildCounts(words) # Build word counts from the input
  try:
    model = hmm.VisibleDataHMM(words, tags, mode == better) # feed data
    model.train(counts) # actually build sigma and tau

    viterbi = decoder.ViterbiDecoder(model, counts)

    # decode the test file:
    for line in testData:
      sentence = FilePreparser.readTestSentence(line)
      yhat = viterbi.decode(sentence)
      tagged = FilePreparser.formatOutput(sentence, yhat)
      outFile.write(tagged+"\n")

  except SyntaxError as e:
    print e
    sys.exit(1)

  _, testWordCount = lm.buildCounts(testData)
  print float(model.unkCount)/testWordCount

  trainFile.close()
  testFile.close()
  outFile.close()

