# Break out some helper functions used by tagger.py into their own little module
#  for increased modularity

from collections import defaultdict

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
  else:
    tags = [str(i) for i in range(0, args.num_tags)]

  return set(tags)

""" Function for building a tagset from a labeled corpus and writing it to a file
      preparser: A preparser inheriting from preparser.AbstractPreparser, initialised with a corpus
      fname: Filename of destination file
    Returns: A set of tags, i.e. a collection of tag classes present in the labeled data
"""
def writeTagsetToFile(preparser, fname):
  tagset = set()

  _, tags = preparser.parseWordsTags()
  for line in tags:
    for tag in line:
      tagset.add(tag)

  f = open(fname, "w")
  for tag in tagset:
    f.write("%s\n"%tag) # one tag per line

  f.close()

