#!/bin/bash

cd ..

# custom config
DATA=/home/qiangwenjie/datasets
TRAINER=ZeroshotCLIP_dense

DATASET=$1
CFG=$2  # config file
save_dir=$3

export CUDA_VISIBLE_DEVICES=0

DIR=output/evaluation/${TRAINER}/${DATASET}/${CFG}
if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}. Skip this job"
else
    echo "Run this job andsave the output to ${DIR}"
    python train_caption.py \
    --root ${DATA} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/zsclip/${CFG}.yaml \
    --output-dir ${DIR} \
    --eval-only \
    TEST.SAVE_PREDS ${save_dir}
fi

# bash zsclip.sh voc2007_distill rn50
# bash zsclip.sh coco2014_distill rn50
# bash zsclip.sh nuswide_distill_limit rn50

# bash zsclip.sh coco2014_distill rn50 tmp_zsclip_coco2014.pkl