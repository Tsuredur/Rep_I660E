#!/bin/bash

DATASET="$1"
SUBSET="$2"
BATCH_SIZE=256
python ./scripts/clip_features_extraction.py -i ./${DATASET}/images/${SUBSET} \
                                             -l ./${DATASET}/${SUBSET}.order \
                                             -d ./${DATASET}/features/clip_features/${SUBSET} \
                                             -b ${BATCH_SIZE}

# if LINUX -->  sed -i 's/\r$//' eval_mmt_bleu.sh