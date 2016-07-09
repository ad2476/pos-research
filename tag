#!/bin/bash

training=$1
testdata=$2
output=$3

# Make TWO copies of this script, one with the original name 'tag' and one
# with the name 'better_tag', corresponding to the two versions of the POS-
# tagger described in the assignment: 
#     'tag' will run the POS-tagger with parameters smoothed by giving
#       *UNK* a pseudo-count of 1;
#     'better_tag' will use the parameters smoothed by setting all words
#       appearing only once in the training corpus to *UNK*.
#   Fill in the line below with whatever sequence of commands will calculate
# the sigmas and taus from the fully labelled corpus $training, and use these
# parameters to tag $testdata and save the result to $output in the _same_
# format as $training and $testdata.
#   Note that $testdata already has tags in it. For this part of the
# assignment, pretend they aren't there (that is, make sure you aren't trying
# to tag tags!).
##############################################################################

##############################################################################
# Run this script with the command
#   ./tag data/wsj2-21.txt data/wsj22.txt output.txt

python2 scripts/tagger.py $training $testdata $output --tag

printf "Score: "
python2 scripts/score.py $testdata $output
