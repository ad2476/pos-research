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

Other branches may be created as development continues to preserve master as stable and allow possibly unstable development on those branches.

## Running the train/test/score routine of the tagger:

Train the tagger, and tag the words from the test corpus. Then, run the scorer on the tagged output, against the "gold" test corpus to evaluate word-level tag accuracy.

For now, the best documentation on how to do this can be found by running `./tag --help` and `./score --help`. Documentation is a WIP.

