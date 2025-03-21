#!/bin/bash

CACHE_HUGGINGFACE="/PATH/TO/CACHE_HUGGINGFACE"
DATA_PATH="/PATH/TO/MMT_DATA/FOLDER"
DATA_MIX_PATH="/PATH/TO/CAPTIONING_DATA/FOLDER"
DUMP_PATH="/RESULTS/SAVING/PATH"
FEAT_PATH="/PATH/TO/MDETR/FEATURES/MMT_DATA,/PATH/TO/CLIP/FEATURES/MMT_DATA"
FEAT_PATH_MIX="/PATH/TO/MDETR/FEATURES/CAPTIONING_DATA,/PATH/TO/CLIP/FEATURES/CAPTIONING_DATA"
MT_MODEL_PATH="/PATH/TO/MT_MODEL_WEIGHTS"
EXP_NAME="EXPERIENCE-NAME"
SRC_LANG="en"
TGT_LANG="fr"
seed="SEED"
EPOCH_SIZE=58000
# SUN, Haoran 2024/11/24
# This work was conducted using SLURM job scheduler.Adapt the scripts to our local configuration.
# The commented part is the original script

#WANDB_MODE=offline NCCL_LL_THRESHOLD=0 python \
#-W ignore \
#-i \
#-m torch.distributed.launch \
#--master_port=9999 \
#--master_addr=${MASTER_ADDR} \
#--nnodes=2 \
#--node_rank=${SLURM_PROCID} \
#--nproc_per_node=4 \
#main.py --beam_size 4 --exp_name ${EXP_NAME} --dump_path ${DUMP_PATH} --features_path ${FEAT_PATH}  --features_type 'mdetr+clip' \
#  --data_path ${DATA_PATH} --data_mix_path ${DATA_MIX_PATH} --features_mix_path ${FEAT_PATH_MIX} --src_lang ${SRC_LANG} --tgt_lang ${TGT_LANG} --dropout '0.4' \
#  --batch_size 16 --optimizer 'adam,lr=0.00001' --epoch_size ${EPOCH_SIZE} --eval_bleu true --max_epoch 5000 --max_len 80 --num_workers 0 \
#  --stopping_criterion "valid_${SRC_LANG}-${TGT_LANG}_mt_bleu,10" --validation_metrics "valid_${SRC_LANG}-${TGT_LANG}_mt_bleu" \
#  --iter_seed ${seed} --other_seed ${seed} --smoothing 0.1 --save_periodic 1 --cache_dir ${CACHE_HUGGINGFACE} --amp 1 --fp16 True \
#  --accumulate_gradients 4 --reload_model ${MT_MODEL_PATH} --multimodal_model --adapters --encoder_attn_mask_text_only --freeze_text_parameters \
#  --start_new_xp_from_ckpt --mix_xp .5 --prob_mask_text 0.25 --encdec_vmlm --guided_self_attention --min_epoch 80 $@

python main.py \
  --exp_name ${EXP_NAME} \
  --dump_path ${DUMP_PATH} \
  --data_path ${DATA_PATH} \
  --src_lang ${SRC_LANG} \
  --tgt_lang ${TGT_LANG} \
  --cache_dir ${CACHE_HUGGINGFACE} \
  --epoch_size ${EPOCH_SIZE} \
  --batch_size 1 \
  --dropout 0.4 \
  --optimizer 'adam,lr=0.00001' \
  --eval_bleu true \
  --max_epoch 5000 \
  --max_len 80 \
  --num_workers 0 \
  --stopping_criterion "valid_${SRC_LANG}-${TGT_LANG}_mt_bleu,10" \
  --validation_metrics "valid_${SRC_LANG}-${TGT_LANG}_mt_bleu" \
  --iter_seed 42 \
  --other_seed 1 \
  --smoothing 0.1 \
  --amp 1 \
  --fp16 True \
  --accumulate_gradients 4 \
  $@
