#!/bin/bash

# This is modeled after get_ende_blue.sh for comparison purposes only!
# https://github.com/tensorflow/tensor2tensor/blob/fc9335c/tensor2tensor/utils/get_ende_bleu.sh
# https://github.com/tensorflow/tensor2tensor/issues/317#issuecomment-380970191

if ! [[ "$MOSESDECODER" ]]; then
  MOSESDECODER="$HOME/mosesdecoder"
fi

DECODED="$1"

if [[ "$2" ]]; then
  TARGET="$2"
else
  TARGET="newstest2013.tok.de"
fi

# Replace unicode.
perl "$MOSESDECODER/scripts/tokenizer/replace-unicode-punctuation.perl" -l de  < "$TARGET" > "$TARGET.norm"
perl "$MOSESDECODER/scripts/tokenizer/replace-unicode-punctuation.perl" -l de  < "$DECODED" > "$DECODED.norm"

# Replace quotations.
PERLIO=:raw perl -ple 's{"}{&quot;}g;' -e 's{„}{&quot;}g;' < "$TARGET.norm" > "$TARGET.quot"
PERLIO=:raw perl -ple 's{"}{&quot;}g;' -e 's{„}{&quot;}g;' < "$DECODED.norm" > "$DECODED.quot"

# Put compounds in ATAT format (comparable to papers like GNMT, ConvS2S).
# See https://nlp.stanford.edu/projects/nmt/ :
# 'Also, for historical reasons, we split compound words, e.g.,
#    "rich-text format" --> rich ##AT##-##AT## text format."'
perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' < "$TARGET.quot" > "$TARGET.atat"
perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' < "$DECODED.quot" > "$DECODED.atat"

# Get BLEU.
perl "$MOSESDECODER/scripts/generic/multi-bleu.perl" "$TARGET.atat" < "$DECODED.atat"
