#!/bin/bash


# custom config
DATA=/workspace/
TRAINER=Caption_distill_double

DATASET=$1
CFG=$2  # config file
CTP=$3  # class token position (end or middle)
NCTX=$4  # number of context tokens
CSC=$5  # class-specific context (False or True)
run_ID=$6
gpu_ID=$7
port=$8



export CUDA_VISIBLE_DEVICES=${gpu_ID}

### 训练 & 测试
for SEED in 1 # 2 3
do
    DIR=./train_output/${run_ID}
    echo "Run this job andsave the output to ${DIR}"
    python -m torch.distributed.launch --nproc_per_node=1 --master_port=${port} --use_env train_caption.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    TRAINER.Caption.N_CTX ${NCTX} \
    TRAINER.Caption.CSC ${CSC} \
    TRAINER.Caption.CLASS_TOKEN_POSITION ${CTP} \
    DATASET.dataset_select 'B' \
    DATASET.caption_feat_root '../output/text_result/'
done

# bash run_all.sh pazhou_distill_chatglm_multi_label_check rn50-1k-ChatGLM_multi_labels_check_diffh_100eEMA end 16 False diffh_100eEMA 0
### chatglm gen
# cd gaoz/visual_prompting_competition/scripts/
# # bash main_mul_check.sh pazhou_distill_chatglm_multi_label_check rn50-1k-ChatGLM_multi_labels_check end 16 False pazhou_chatglm_16mpt_soft_hard_challenge 0
# bash main_mul_check.sh pazhou_distill_chatglm_multi_label_check rn50-1k-ChatGLM_multi_labels_check end 16 False pazhou_chatglm_16mptB_softhard_20e 0
# bash main_mul_check.sh pazhou_distill_chatglm_multi_label_check rn50-1k-ChatGLM_multi_labels_check end 16 False pazhou_chatglm_16mptB_softhard_20eEMA 0
# bash main_mul_check.sh pazhou_distill_chatglm_multi_label_check rn50-1k-ChatGLM_multi_labels_check end 16 False pazhou_chatglm_16mptB_totalhard_100e 0
# bash main_mul_check.sh pazhou_distill_chatglm_multi_label_check rn50-1k-ChatGLM_multi_labels_check end 16 False pazhou_chatglm_16mptB_totalhard_20eEMA 0
# bash main_mul_check.sh pazhou_distill_chatglm_multi_label_check rn50-1k-ChatGLM_multi_labels_check end 16 False pazhou_chatglm_16mptB_totalmore_20e 0
# bash main_mul_check.sh pazhou_distill_chatglm_multi_label_check rn50-1k-ChatGLM_multi_labels_check end 16 False pazhou_chatglm_16mptB_softmoreEMA_100e 0
# bash main_mul_check.sh pazhou_distill_chatglm_multi_label_mix rn50-1k-ChatGLM_multi_labels_mix end 16 False pazhou_chatglm_16mptB_zuanEMA_100e 0
# bash main_mul_check.sh pazhou_distill_chatglm_multi_label_check rn50-1k-ChatGLM_multi_labels_check end 16 False pazhou_chatglm_16mptB_zuantotalmore_100eEMA 0