# -*- coding: utf-8 -*-

from rules import listToRuleset

class AbstractUnker:
  """ Initialise the class with:
        corpus: a list of sentences - each sentence a list of words - the corpus to substitute UNKs for
        counts: a dictionary mapping word->count
        unk_thresh: threshold count for when a word should be UNKed
                    i.e. only substitute UNK for words with count<=unk_thresh

      The corpus is processed in this __init__ method, which will take approx. O(n) time
       for n words in the corpus.
  """
  def __init__(self, corpus, counts, unk_thresh=1):
    self._corpus = corpus
    self._counts = counts
    self._thresh = unk_thresh

    # _abstractGuard() should init self._rulesList in subclasses
    self._rulesList = None

    self._unkedCorpus = None # initialised in _processCorpus()

    self._abstractGuard() # raise error only if self is AbstractUnker but not a subclass
    self._rulesetHead = listToRuleset(self._rulesList)
    self._processCorpus()


  """ OVERRIDE THIS IN ALL SUBCLASSES!! """
  def _abstractGuard(self):
    raise NotImplementedError('Cannot instantiate AbstractUnker')

  """ Given a word that should be substituted for UNK, evaluate
      which category of UNK applies based on this Unker's rules
  """
  def _categoriseUnk(self, word):
    ruleset = self._rulesetHead # get the list of rules we need to apply

    # while there are still rules to evaluate:
    unk = "*U*" # catchall UNK category
    while ruleset is not None:
      rule = ruleset.getRule()
      if rule.appliesTo(word):
        unk = rule.execute()
        break # stop at first applicable UNK substitution

      ruleset = ruleset.nextRule() # otherwise, keep looking

    return unk

  """ Process the corpus, internally store the corpus w. UNK substitutions """
  def _processCorpus(self):
    res = [] # will be a copy of the original corpus but with any UNK substitutions

    for line in self._corpus:
      newl = [] # the new line with UNK substituted

      for word in line:
        # substitute UNK only if below threshold
        if self._counts[word] <= self._thresh:
          before = word
          word = self._categoriseUnk(word)

        newl.append(word) # append this word (poss. with sub.)

      res.append(newl) # append this line to the resulting list

    self._unkedCorpus = res

  """ Check if a word is not in the dictionary (i.e. it is unknown)
       and return the properly Unked form. Otherwise return the word.
  """
  def evaluateWord(self, word):
    if word not in self._counts:
      return self._categoriseUnk(word)

    return word

  """ Return the corpus with UNKs substituted as necessary """
  def getUnkedCorpus(self):
    return self._unkedCorpus

  """ Return the original word found in the jth position of the ith
       sentence of the unmodified corpus.
       e.g. getOrigWord(0,1) will return "world"
            for corpus=[["hello", "world"],["demo", "example"]]
  """
  def getOrigWord(self, i, j):
    return self._corpus[i][j]

  """ Return the original corpus without any UNK substitutions """
  def getOrigCorpus(self):
    return self._corpus

""" Unker for Sanskrit that uses a rudimentary knowledge of Sanskrit grammar
    to place low-count words into UNK categories based on commonly-seen morphologies

    The corpus given should contain paṭhapāda Sanskrit text transliterated to IAST
"""
class PratyayaUnker(AbstractUnker):

  """ Overriding allows class to be instantiated. Also allows the custom set
       of rules to be specified on a per-class basis.
  """
  def _abstractGuard(self):
    _tavya, Utavya = "tavya", "*Utavya*" # gerundive e.g. kartavya
    _nIya, _NIya, UnIya = "n\xc4\xabya", "\xe1\xb9\x87\xc4\xabya", "*UnIya*" # gerundive e.g. karaṇīya
    _asya, Uasya = "asya", "*Uasya*" # 6. vibhakti sing. -a stem
    _ena, _eNa, Uena = "ena", "e\xe1\xb9\x87a", "*Uena*" # 3. vibhakti sing. -a stem
    _Su, _su, Usu = "\xe1\xb9\xa3u", "su", "*Usu*" # 7. vibhakti plu. -su/-ṣu represent the same form
    _as, Uas = "a\xe1\xb8\xa5", "*Uas*" # 1s/1p/2p/5s/6s/etc -aḥ is a common inflectional ending
    _As, UAs = "\xc4\x81\xe1\xb8\xa5", "*UAs*" # 1p/etc -āḥ also a common inflectional ending
    _am, _aM, Uam = "am", "am\xcc\xa3", "*Uam*" # 1s/2s
    _Am, _AM, UAm = "\xc4\x81m", "\xc4\x81\xe1\xb9\x83", "*UAm*" # 2s/6p/etc
    _au, Uau = "au", "*Uau*" # 1d/2d/etc
    _is, Uis = "i\xe1\xb8\xa5", "*Uis*"
    _os, Uos = "o\xe1\xb8\xa5", "*Uos*"
    _At, UAt = "\xc4\x81t", "*UAt*" # 5. vibhakti sing. -a stem
    _at, Uat = "at", "*Uat*" # mostly seen in parasmai-/ātmanepada 3s verb forms?
    _vA, _tyA, UvA = "v\xc4\x81", "ty\xc4\x81", "*UvA*" # gerunds often end in -vā or -tyā
    _A, UA = "\xc4\x81", "*UA*" # e.g. kanyā, rājā, etc.
    _e, Ue = "e", "*Ue*" # 7. vibh. sing. -a stem, or vocatives etc
    _s, Us = "\xe1\xb8\xa5", "*Us*" # consonant stem ending -ḥ
    _i, Ui = "i", "*Ui*" # 7. vibh. sing.
    _I, UI = "\xc4\xab", "*UI*" # 1. vibh. sing. fem.

    wordEnds = lambda w,s: w[-len(s):] == s

    self._rulesList = [(lambda w: wordEnds(w, _tavya), Utavya),
                       (lambda w: wordEnds(w, _nIya) or wordEnds(w, _NIya), UnIya),
                       (lambda w: wordEnds(w, _asya), Uasya),
                       (lambda w: wordEnds(w, _ena) or wordEnds(w, _eNa), Uena),
                       (lambda w: wordEnds(w, _su) or wordEnds(w, _Su), Usu),
                       (lambda w: wordEnds(w, _as), Uas),
                       (lambda w: wordEnds(w, _As), UAs),
                       (lambda w: wordEnds(w, _am) or wordEnds(w, _aM), Uam),
                       (lambda w: wordEnds(w, _Am) or wordEnds(w, _AM), UAm),
                       (lambda w: wordEnds(w, _au), Uau),
                       (lambda w: wordEnds(w, _is), Uis),
                       (lambda w: wordEnds(w, _os), Uos),
                       (lambda w: wordEnds(w, _At), UAt),
                       (lambda w: wordEnds(w, _at), Uat),
                       (lambda w: wordEnds(w, _vA) or wordEnds(w, _tyA), UvA),
                       (lambda w: wordEnds(w, _A), UA),
                       (lambda w: wordEnds(w, _e), Ue),
                       (lambda w: wordEnds(w, _s), Us),
                       (lambda w: wordEnds(w, _i), Ui),
                       (lambda w: wordEnds(w, _I), UI)]

""" A basic Unker class that uses only one UNK category. Also a reference
    implementation for AbstractUnker.
"""
class BasicUnker(AbstractUnker):

  """ Overriding allows class to be instantiated. Also allows the custom set
       of rules to be specified on a per-class basis.
  """
  def _abstractGuard(self):
    # technically this is redundant, since categoriseUnk() will default to "*U*" if
    #  no other categories apply
    self._rulesList = [(lambda w: True, "*U*")]

