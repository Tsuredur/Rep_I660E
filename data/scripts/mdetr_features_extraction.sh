#!/bin/bash

DATASET="$1"
SUBSET="$2"
TORCH_HUB_DIR="/PATH/TO/HUB/DIR"
python ./scripts/mdetr_features_extraction.py \
  -i ./${DATASET}/images/${SUBSET} \
  -d ./${DATASET}/features/mdetr_features/${SUBSET} \
  -l ./${DATASET}/${SUBSET}.order \
  --text ./${DATASET}/${SUBSET}.en \
  --threshold 0.2 \
  --hub_dir ${TORCH_HUB_DIR}

# if LINUX -->  sed -i 's/\r$//' eval_mmt_bleu.sh