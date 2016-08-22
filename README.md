Part-of-Speech Tagging in Sanskrit using HMM
===

Part-of-speech (POS) tagging, within computational linguistics, aims to use computational methods todiscern what part of speech (i.e. noun, verb, adjective) each word in a document corresponds to. This process, while non­trivial, has been extensively researched using statistical and rule­based tagging systems within the context of English and other European languages. However, less research exists aimed at highly morphological languages such as Sanskrit. Unique difficulties present themselves in the application of English-based POS-tagging methods to the Sanskrit language.

This research project aims to address a foremost issue with POS tagging for Sanskrit: a small amount of labeled data, combined with a need for a larger tagset (as compared to English) results in a relatively low word- and sentence-level tagging accuracy. Building off of tagged data and a tagset developed by researchers R. Chandrashekar and Girish Nath Jha at JNU[\[1\]](http://sanskrit.jnu.ac.in/corpora/tagset.jsp), this project shall train a Hidden Markov Model in a semi-supervised fashion by making use of both the JNU tagged corpus as well as the large quantities of digitised Sanskrit text available from [GRETIL](http://gretil.sub.uni-goettingen.de/).

What follows in this README is a cursory User's Guide of sorts for this project. For detailed documentation on the research (background, model, notation, implementation, methods, results) please see `doc/`. Much of this documentation is still forthcoming.

## About this repository:

The `master` branch of this repository represents the most stable iteration of the model. As of the latest update to this README, this is a (relatively) stable implementation of an HMM trained solely from visible data.

The `devel` branch is less stable and contains possibly breaking changes that will eventually be merged into master once functioning.

### Directory structure:

```
.
├── data: corpora, tagged output files
│   ├── en: English corpora
│   └── sans: Sanskrit corpora
├── debug: impermanent output logs
├── doc: documentation
├── perfstats: performance statistics of the model under various parameters
├── pos: python modules
│   └── hmm: modules specific to HMM implementation
└── tools: various utility modules and/or scripts
```

## Dependencies:

The required dependencies to build and use the pos module are:
* Python >= 2.6. Tested on 2.7.11. Not currently compatible with Python3.
* NumPy >= 1.10. Tested on 1.11.0.
* Cython >= 0.24.

The process to install the above requirements varies based on your system. Most should be available through your package manager's repositories.

This project currently expects a Unix-like system. It has been tested to run, with the required dependencies correctly configured, on OS X 10.10, Arch Linux, Debian and Fedora, and should be able to run on other Linuxes without much hassle (perhaps a tweak in the makefile at most). If for some reason the `hmm` module cannot be built, or the tagger/related tools do not run, consider that a bug. No support is expected for Windows, although there may be some success running within the new WSL (not tested).

## Building:

As distributed by this repository, the `hmm.HiddenDataHMM` module must first be built before use. From within the `pos/hmm/` directory, run `make` (`make -f makefile-osx` on OS X).

It might be necessary, in order for certain bash scripts to work out-of-the-box, to allow execute permissions on `tagger.py`, `score.py`, and `tools/corpus.py`. Naturally, the bash scripts themselves should be executable as well.

## Generating a corpus:

### About the data:
Labeled and unlabeled data in English and Sanskrit are found under `data/en/` and `data/sans/`, respectively. A large (~39k line) corpus of tagged English sentences exists in the form of `wsj2-21.txt` and `wsj22.txt` from the Penn Treebank; its unlabeled equivalent is `wsj2-21-notags.txt`. For training an HMM from labeled data (for English), `wsj2-21.txt` can be used for training and `wsj22.txt` for testing. A (substantially) smaller corpus of 174 labeled English sentences exists as `TaggedCorpus.txt` (under `data/en/`, along with all other English corpora). The reason for the small corpus is to provide a point of comparison to the performance of the tagger on Sanskrit.

Sanskrit labeled data exists as `TaggedCorpus.txt` (under `data/sans/`). This is a corpus of 174 lines of labeled Sanskrit, from Dr. Chandrashekar's JNU corpus (with modifications to the tags, see `doc/tagset_guidelines.md`). The file `GRETILNoTagsTrain.txt` is a corpus of ~3.1k lines of unlabeled padapāṭha (sandhi-separated) Sanskrit sentences, concatenated from the following texts:
* Śivamahimnastava (`sivamahinastava.txt`), text sourced from [GRETIL](http://gretil.sub.uni-goettingen.de/gretil/1_sanskr/4_rellit/saiva/sivmstau.htm)
* Stavacintamāṇi of Bhaṭṭa Nārāyaṇa (`bhatta_narayana.txt`), text sourced from [GRETIL](http://gretil.sub.uni-goettingen.de/gretil/1_sanskr/4_rellit/saiva/bhnstcxu.htm)
* Kirātārjunīya (`kiratarjuniya.txt`), text sourced from [GRETIL](http://gretil.sub.uni-goettingen.de/gretil/1_sanskr/5_poetry/2_kavya/bhakirxu.htm)
* Rāmacaritamahākavya (`ramacaritamahakavya.txt`), text sourced from [GRETIL](http://gretil.sub.uni-goettingen.de/gretil/1_sanskr/5_poetry/2_kavya/rmc1-3xu.htm)

### Splitting labeled data into test and train corpora:
The `corpus.py` utility (under `tools/`) is used to split a labeled corpus into both training and testing files. The user can specify to select either a single line for use as a test corpus, and the remainder for training (useful for cross-validation); or they can specify a ratio of test/train sizes; or lastly a specific number of lines to use as a test corpus, with the remainder for training. If a ratio or line count is specified, the actual lines will be selected at random.

`tools/corpus.py --help` should provide a pretty good sense of how this utility works. For example, `tools/corpus.py --ratio 0.2 data/sans/TaggedCorpus.txt data/sans` will generate `data/sans/train.txt` and `data/sans/test.txt`, with `test.txt` containing approx. 20% the lines of `TaggedCorpus.txt` and `train.txt` comprising the remaining ~80%.

## Running the tagger:

Use `tagger.py` to train a Hidden Markov model, and decode a document using the Viterbi method. Make sure `tagger.py` is set as executable.

Tagging can be done three ways: on labeled data, on unlabeled data, or on both. These are specified via the `--model` option. Pass to the tagger a training corpus, which can be a list of files; it should be a labeled corpus for the supervised and semi-supervised models, and unlabeled for the unsupervised model. Pass also a test corpus, which can be a list of files, all of which must be labeled. If using the semi-supervised model, an extra unlabeled corpus must be specified the `--extra` flag. Specify a location for tagged output with `--output`. Lastly, specify what language the tagger should run on (currently either English or Sanskrit).
For more information on how to run the tagger, including additional flag options, run `./tagger.py --help`.

### Examples:
Tagging in English, on labeled data:
```
$ ./tagger.py --lang EN --model super --train data/en/wsj2-21.txt --test data/en/wsj22.txt --output data/output.txt
```
Tagging in Sanskrit, on unlabeled data, for 184 tag classes and 10 iterations of Baum-Welch:
```
$ ./tagger.py --lang SANS --model unsuper --iter 10 -n 184 --train data/sans/GRETILNoTagsTrain.txt \
   --test data/sans/test.txt --output data/output.txt
```
Tagging in Sanskrit on both labeled and unlabeled data:
```
$ ./tagger.py --lang SANS --model semisuper --iter 5 --train data/sans/train.txt --test data/sans/test.txt \
  --extra data/sans/GRETILNoTagsTrain.txt --output data/output.txt
```
For generating the files `data/sans/train.txt` and `data/sans/test.txt`, see `tools/corpus.py --help` and/or the previous section.

## Scoring (aka does this thing work?):

Upon the tagger completing, there should be a file (specified by `--output`, e.g. `data/output.txt`) containing its tagged output sentences. Use `score.py` to compare this against the "gold" file, which is usually what was passed as `--test` to the tagger.

Run `./score.py --help` for a detailed overview of the options. Note that, in lieu of a tagset file, an empty file (e.g. `/dev/null`) can be passed if balanced accuracy or a printout of the confusion matrix is not desired. Also pass an empty file if the classes of the confusion matrix should be only the classes present in the gold and tagged files, rather than all valid classes.

### About the `crossvalidate` script:
When evaluating performance of the model on Sanskrit (or other small labeled corpora, c.f. `data/en/TaggedCorpus.txt`), cross-validation is necessary to properly assess how well it might generalise to unknown data without compromising model accuracy by reducing available training data even further. The `crossvalidate` script automates the process.

```
Usage: ./crossvalidate tagger_args [...]
       tagger_args: the set of arguments passed to ./tagger.py, excluding --test --train --output and --lang,
                    which are specified by this script
```
The script will default to working on Sanskrit, but this behaviour can be changed by setting `MODEL_LANG=EN` as an environment variable, or by editing it in the script.

### About the `eval` script:
In order to assess the performance of the semi-supervised model under increasing amounts of unlabeled data, the `eval` script automates the process of creating an unlabeled corpus of increasing sizes and training the tagger with that.

```
usage: eval STEP_SIZE EM_ITER OUTPUT (EN|SANS)
Evaluate accuracy of semi-supervised HMM as a function of unlabeled corpus size.

Arguments:
  STEP_SIZE: Num. of lines data is increased by on each iteration.
  EM_ITER: Num. of iterations of EM
  OUTPUT: Path to output table of corpus size vs. accuracy
  (EN|SANS): Either EN (english) or SANS (sanskrit). Defaults to English.
```
Run `./eval` without any parameters to print the above help.
