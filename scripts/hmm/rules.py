
""" The Unker classes evaluate on what basis to UNK a word, and may specify
    multiple UNK categories. An UnkRule stores the condition for a word to
    be classed into an UNK category i.e. the result of the rule
"""
class UnkRule:

  """ condition: a function f::String->Bool that takes a word and
                 evaluates if it passes some boolean test
      result: the result of applying this rule to a word (assuming word meets condition)
  """
  def __init__(self, condition, result):
    self._cond = condition
    self._res = result

  """ Checks if the rule applies to the given word
      e.g. return True if condition applies to word, False if not
  """
  def appliesTo(self, word):
    return self._cond(word)

  """ Executes the rule i.e. returns the result of the rule """
  def execute(self):
    return self._res

""" Essentially a linked list for UnkRules """
class UnkRuleset:

  """ rule: an UnkRule
      nextRule: an UnkRuleset or None
  """
  def __init__(self, rule, nextRule):
    self._head = rule
    self._tail = nextRule
  
  def getRule(self):
    return self._head

  def nextRule(self):
    return self._tail
  
""" Convert a list of tuples into a linked-list UnkRuleset
    Each tuple should be in the form (condition, result). This
    should allow for fewer imports required by an importing module
"""
def listToRuleset(rules):
  nextRule = None
  ruleset = None
  for ruleTuple in reversed(rules):
    cond,res = ruleTuple # deconstruct the tuple

    rule = UnkRule(cond,res) # construct an UnkRule
    ruleset = UnkRuleset(rule, nextRule)
    nextRule = ruleset

  return ruleset

