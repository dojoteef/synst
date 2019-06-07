#!/bin/bash

BPE_FILE="$1"
if ! [[ "$BPE_FILE" ]]; then
  echo "Must specifiy path to BPE file"
  exit 1
fi

OUT_FILE="${BPE_FILE/.bpe.32000/}"
if ! [[ -f "$OUT_FILE" ]]; then
  sed -r 's/(@@ )|(@@ ?$)//g' < "$BPE_FILE" > "$OUT_FILE"
fi

SCRIPT_PATH="$(dirname "$0")"
PARSE_FILE="$OUT_FILE.parse"
if ! [[ -f "$PARSE_FILE" ]]; then
  "$SCRIPT_PATH/parse.sh" "$OUT_FILE"
fi
