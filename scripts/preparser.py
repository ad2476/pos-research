# -*- coding: utf-8 -*-
import re
import itertools
from hmm import STOP

""" This class will pre-parse POS-tagged files in the format of the WSJ data """
class EnglishWSJParser:

  def __init__(self, inputData):
    self._rawdata = inputData
    self._stopPair = STOP + " " + STOP

  """ Parse the outputs and tags into separate lists """
  def parseWordsTags(self):
    words = [STOP]
    tags = [STOP]

    for line in self._rawdata:
      line = "%s %s" % (line, self._stopPair)
      sentence = line.split()
      # i increments by 2 from 0 to len(sentence)
      for i in xrange(0, len(sentence), 2):
        word = sentence[i]
        label = sentence[i+1]

        words.append(word)
        tags.append(label)

    return words,tags

  def readTestSentence(self, line):
    return line.split()[::2]

  def formatOutput(self, words, tags):
    output = ""
    for word,tag in itertools.izip(words, tags):
      output += word + " " + tag + " "

    return output

class SanskritJNUParser:
  
  def __init__(self, inputData):
    self._rawdata = inputData
    self._stopPair = "%s[%s]" % (STOP, STOP)

  def parseWordsTags(self):
    words = [STOP]
    tags = [STOP]

    for line in self._rawdata:
      line = "%s %s" % (line, self._stopPair)
      # each line is formatted: "WORD[TAG] WORD[TAG] WORD[TAG]DANDA[TAG]\n"
      
      # capturing group before a literal '[' char:
      #  match at least 0 times,lazy, on a group consisting of:
      #   not whitespace, not a ']' char
      words.extend(re.findall(r"([^\s\]]*?)\[", line))

      tags.extend(re.findall(r"\[(.*?)\]", line)) # find everything within brackets

    return words,tags

  def readTestSentence(self, line):
    return re.findall(r"([^\s\]]*?)\[", line)

  def formatOutput(self, words, tags):
    n = len(words)
    output = ""
    for i in xrange(0, n):
      output += words[i] + "[" + tags[i] + "] "

    return output

    
