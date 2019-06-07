#!/bin/bash

# Requires sacrebleu in your PATH
if ! [[ "$MOSESDECODER" ]]; then
  MOSESDECODER="$HOME/mosesdecoder"
fi

DECODED="$1"
if ! [[ "$2" ]]; then
  SOURCE_LANGUAGE="en"
else
  SOURCE_LANGUAGE="$2"
fi

if ! [[ "$3" ]]; then
  TARGET_LANGUAGE="de"
else
  TARGET_LANGUAGE="$3"
fi
shift 3

# Detokenize
perl "$MOSESDECODER/scripts/tokenizer/detokenizer.perl" -l "$TARGET_LANGUAGE"  < "$DECODED" > "$DECODED.detok"

PARAMS=("-tok" "intl" "-l" "$SOURCE_LANGUAGE-$TARGET_LANGUAGE")
if [[ -z "$#" ]]; then
  # No target provided, so pick a default one...
  PARAMS+=("-t" "wmt13")
elif [[ "$#" -gt 1 ]] || [[ -f "$1" ]]; then
  # Must be a list of references
  PARAMS+=("$@")
else
  # If the target is not a reference file then it must be a named test set
  PARAMS+=("-t" "$1")
fi

sacrebleu "${PARAMS[@]}" < "$DECODED.detok"
