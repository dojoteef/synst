#!/bin/bash

INPUT=$1
CONFIG_PATH="$(dirname "$0")"
PARAMS=( \
  "-Dorg.slf4j.simpleLogger.defaultLogLevel=warn" \
  "edu.stanford.nlp.pipeline.StanfordCoreNLP" \
  )
if [[ "$INPUT" == *.en ]]; then
  PARAMS+=("-props" "$CONFIG_PATH/corenlp-english.properties")
elif [[ "$INPUT" == *.fr ]]; then
  PARAMS+=("-props" "$CONFIG_PATH/corenlp-french.properties")
elif [[ "$INPUT" == *.de ]]; then
  PARAMS+=("-props" "$CONFIG_PATH/corenlp-german.properties")
fi

if ! [[ "$(which java)" ]]; then
  PATH="$PATH:/usr/lib/jvm/java-8-openjdk-amd64/bin"
fi

if ! [[ "$CLASSPATH" ]]; then
  CLASSPATH="$HOME/devel/third_party/stanford-corenlp/*"
fi
export CLASSPATH

# Awk script to parse the latex tree format into a linearized form comprised of
# depth/token pairs, i.e. converts:
# .[
# ROOT
#   NP
#     A
#   PUNC
#     .
# .]
#
# into: 0 ROOT 1 NP 2 A 1 PUNC 2 .

PARSE_FILE="$INPUT.parse"
# shellcheck disable=SC2016
AWK_SCRIPT='/^.\[$|^.\]$/{if($1==".["){next}{print "";next}}{printf "%d %s ", gsub(" ","")/2,$1}'
java "${PARAMS[@]}" < "$INPUT" | jq -r .sentences[].parse | \
  awk "$AWK_SCRIPT" > "$PARSE_FILE.incomplete"
mv "$PARSE_FILE.incomplete" "$PARSE_FILE"
