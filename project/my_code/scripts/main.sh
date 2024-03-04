#!/bin/bash

cd ..

# custom config
DATA=./
TRAINER=Caption_distill_double_ema

DATASET=$1
CFG=$2  # config file
CTP=$3  # class token position (end or middle)
NCTX=$4  # number of context tokens
CSC=$5  # class-specific context (False or True)
run_ID=$6


export CUDA_VISIBLE_DEVICES=$7

### 训练 & 测试
for SEED in 1 # 2 3
do
    DIR=output/${run_ID}/${TRAINER}/${CFG}/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
    echo "Run this job andsave the output to ${DIR}"
    python train_caption.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    TRAINER.Caption.N_CTX ${NCTX} \
    TRAINER.Caption.CSC ${CSC} \
    TRAINER.Caption.CLASS_TOKEN_POSITION ${CTP} \
    TRAIN.adapter False \
    TRAIN.ema False
done

### 从现有的训练模型运行测试
# for SEED in 1
# do
#     DIR=output/evaluation/${run_ID}/${TRAINER}/${CFG}/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
#     echo "Run this job andsave the output to ${DIR}"
#     python train_caption.py \
#     --root ${DATA} \
#     --seed ${SEED} \
#     --trainer ${TRAINER} \
#     --dataset-config-file configs/datasets/${DATASET}.yaml \
#     --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
#     --output-dir ${DIR}\
#     --model-dir output/${run_ID}/${TRAINER}/${CFG}/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}\
#     --eval-only \
#     TRAINER.Caption.N_CTX ${NCTX} \
#     TRAINER.Caption.CSC ${CSC} \
#     TRAINER.Caption.CLASS_TOKEN_POSITION ${CTP}
#     # fi
# done

# bash main.sh pazhou_distill_chatglm_multi_label_mix rn50-1k-ChatGLM_multi_labels_mix end 64 False pazhou_chatglm_64mpt_mix_sample 0



