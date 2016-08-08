import sys
import argparse
from collections import defaultdict

import hmm
import decoder
import preparser

DFLT_ITER_CAP = 1

def parseProgramArgs():
  parser = argparse.ArgumentParser(description="HMM-based part-of-speech tagger. See README.md for detailed documentation")
  group1 = parser.add_argument_group("Data", "Specify input data to the tagger.")
  group1.add_argument("--train", nargs='+', required=True,
                      help="Path(s) to training corpora. Format should match model type.")
  group1.add_argument("--test", nargs='+', required=True,
                      help="Path(s) to test corpora.")
  group1.add_argument("--output", required=True, help="Destination path for tagged test output")
  group1.add_argument("--extra", nargs='+', help="Path to supplementary unlabeled corpus. Required if using semi-supervised, ignored otherwise.")

  group2 = parser.add_argument_group("Model types", "Define the model being used.")
  group2.add_argument("--lang", choices=["EN", "SANS"], required=True,
                      help="Select tagging language.")
  group2.add_argument("--model", choices=["super", "unsuper", "semisuper"], required=True,
                      help="Train HMM in supervised, unsupervised or semi-supervised fashion.")

  group3 = parser.add_argument_group("Meta-parameters", "Tweak meta-parameters to the model.")
  group3.add_argument("--iter", type=int, help="Specify number of iterations of EM (for semi- and unsupervised models). Omit for 1 iteration default.")
  group3_mutex = group3.add_mutually_exclusive_group()
  group3_mutex.add_argument("--tagfile", help="Path to file containing a tagset.")
  group3_mutex.add_argument("-n", "--num_tags", type=int, default=1,
                            help="Number of tags the unsupervised model should use.")

  return parser.parse_args()

""" Build unigram counts from a corpus, aka n_w(d)
    Inputs:
      - document: An input corpus to build counts for
    Returns:
      - A tuple of counts (str->int) aka n_w(d)
"""
def buildCounts(document):
  counts = defaultdict(int) # map string -> int: aka n_w(d)

  # Populate n_w(d) aka counts dict
  n_o = 0
  for line in document:
    words = line
    for word in words:
      word = hash(word)
      counts[word] += 1
      n_o += 1

  return counts, n_o

""" Given a list of filenames, concatenate contents into a list of sentences.
    Each sentence is a string.
"""
def buildCorpus(files):
  corpus = []
  for fname in files:
    f = open(fname, 'r')
    corpus.extend([line for line in f])
    f.close()

  return corpus

""" Build a tagset, given relevant cmdline args """
def buildTags(args):
  if args.tagfile:
    tagfile = open(args.tagfile, 'r') # this is a file with tags separated by whitespace
    tags = []
    for line in tagfile:
      tags.extend(line.split())
    tagfile.close()
    tags = set(tags)
  else:
    tags = set([str(i) for i in range(0, args.num_tags)])

  return tags

if __name__ == '__main__':

  args = parseProgramArgs()
  iter_cap = args.iter or DFLT_ITER_CAP

  trainData = buildCorpus(args.train)
  testData = buildCorpus(args.test)

  outFile = open(args.output, 'w')

  # Determine which preparser to use (this can be extensible)
  if args.lang == "EN":
    FilePreparser = preparser.EnglishWSJParser
  elif args.lang == "SANS":
    FilePreparser = preparser.SanskritJNUParser

  # Set up models depending on the type:
  if args.model == "super":
    data = FilePreparser(trainData).parseWordsTags()
    if data is None:
      sys.stderr.write("Error parsing input: Bad format.\n")
      sys.exit(1)

    words, tags = data
    counts,wc = buildCounts(words) # Build word counts from the input
    model = hmm.VisibleDataHMM(words, tags, counts, wc)
    params = None
  elif args.model == "unsuper":
    words = FilePreparser(trainData).parseWords()

    counts,wc = buildCounts(words)
    tagset = buildTags(args)
    model = hmm.HiddenDataHMM(words, tagset, wc)
    params = (iter_cap, None)
  else: # model is semi-supervised
    data = FilePreparser(trainData).parseWordsTags()
    if data is None:
      sys.stderr.write("Error parsing input: Bad format.\n")
      sys.exit(1)

    words, tags = data
    counts,wc = buildCounts(words)
    visibleModel = hmm.VisibleDataHMM(words, tags, counts, wc) # now we have a visible model
    visibleModel.train() # build the counts from the visible model

    params = (iter_cap, (visibleModel.getDistribution(), visibleModel.getVisibleCounts()))
    if not args.extra: # make sure the user has specified this option
      sys.stderr.write("--extra must be specified if --model=semisuper\n")
      sys.exit(1)

    unlabeledData = buildCorpus(args.extra)
    extraWords = FilePreparser(unlabeledData).parseWords() # preparse unlabeled data
    if extraWords is None:
      sys.stderr.write("Error parsing extra input: Bad format.\n")
      sys.exit(1)
    _,ewc = buildCounts(extraWords) # build counts from the extra data
    wc += ewc # add on total word count from extraWords to first wc
    tagset = visibleModel.getLabels() # get the tags from visible data
    # pass along the label hash from the visible model to our hidden model:
    model = hmm.HiddenDataHMM(extraWords, tagset, wc, visibleModel.getLabelHash())

  model.train(params) # train our model with the given training parameters

  viterbi = decoder.ViterbiDecoder(model, counts)

  # decode the test file:
  prep = FilePreparser(testData)
  for line in testData:
    sentence = prep.getSentenceWords(line)
    yhat = viterbi.decode(sentence)
    tagged = prep.formatOutput(sentence, yhat)
    outFile.write(tagged+"\n")

  outFile.close()

