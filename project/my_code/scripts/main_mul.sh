#!/bin/bash

cd ..

# custom config
DATA=./
TRAINER=Caption_distill_double

DATASET=$1
CFG=$2  # config file
CTP=$3  # class token position (end or middle)
NCTX=$4  # number of context tokens
CSC=$5  # class-specific context (False or True)
run_ID=$6


export CUDA_VISIBLE_DEVICES=0,1

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
    TRAINER.Caption.CLASS_TOKEN_POSITION ${CTP} # \
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



### chatglm gen
# bash main.sh pazhou_distill_chatglm rn50 end 16 False pazhou_chatglm_valid 0
