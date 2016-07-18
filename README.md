Part-of-Speech Tagging in Sanskrit using HMM
===

Part-of-speech (POS) tagging, within computational linguistics, aims to use computational methods to 
discern what part of speech (i.e. noun, verb, adjective) each word in a document corresponds to. This 
process, while non­trivial, has been extensively researched using statistical and rule­based tagging 
systems within the context of English and other European languages. However, less research exists aimed 
at highly morphological languages such as Sanskrit. Unique difficulties present themselves in the 
application of English-based POS-tagging methods to the Sanskrit language.

This research project aims to address a foremost issue with POS tagging for Sanskrit: a small amount of labeled data, combined with a need for a larger tagset (as compared to English) results in a relatively low word- and sentence-level tagging accuracy. Building off of tagged data and a tagset developed by researchers R. Chandrashekar and Girish Nath Jha at JNU[\[1\]](http://sanskrit.jnu.ac.in/corpora/tagset.jsp), this project shall train a Hidden Markov Model in a semi-supervised fashion by making use of both the JNU tagged corpus as well as the large quantities of digitised Sanskrit text available from [GRETIL](http://gretil.sub.uni-goettingen.de/).

## Development:

The `master` branch of this repository represents the most stable iteration of the model. As of the latest update to this README, this is a (relatively) stable implementation of an HMM trained solely from visible data.

The branch `em-devel` contains more current development on implementing EM (Baum-Welch) for HMMs.

## Running the train/test/score routine of the tagger:

Two scripts: `tag` and `better_tag`. The former does regular tagging, the latter will replace all words with count of 1 in the training corpus with `*UNK*`. `better_tag` is recommended for higher accuracy.

To run on Sanskrit, make sure the tagger and scorer are using the `SanskritJNUParser` preparsing class. For English, on the Wall Street Journal corpora, use `EnglishWSJParser` class. This requires editing a single line in tagger.py and another single line in score.py.

### Example: Running the script:

`./better_tag data/wsj2-21.txt data/wsj22.txt data/tagged.txt` will train and evaluate the model on the Wall Street Journal corpus.

