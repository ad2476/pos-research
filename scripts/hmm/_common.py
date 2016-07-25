
"""Map the association between a pos label and its integer index.
   Necessary because hmms store labels as ints for faster indexing on numpy arrays,
    but there needs to be a way to know what label maps to what internal index.
"""
def makeLabelHash(labels):
 labelHash = {}
 for i,y in enumerate(labels):
  labelHash[y] = i

 return labelHash

