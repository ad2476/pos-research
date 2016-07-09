Part-of-Speech Tagging in Sanskrit using HMM
===

Two scripts: `tag` and `better_tag`. The former does regular tagging, the latter will replace all words with count of 1 in the training corpus with `*UNK*`. `better_tag` is recommended for higher accuracy.

To run on Sanskrit, make sure the tagger and scorer are using the `SanskritJNUParser` preparsing class. For English, on the Wall Street Journal corpora, use `EnglishWSJParser` class. This requires editing a single line in tagger.py and another single line in score.py.

# Example: Running the script:

`./better_tag data/wsj2-21.txt data/wsj22.txt data/tagged.txt` will train and evaluate the model on the Wall Street Journal corpus.

