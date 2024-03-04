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

# use evidence and freq
for SEED in 1 # 2 3
do
    DIR=./train_output_b_more_patchs_5/evaluation
    echo "Run this job andsave the output to ${DIR}"
    python -m torch.distributed.launch --nproc_per_node=1 --master_port=${port} --use_env train_caption.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir ./best_model \
    --eval-only \
    TRAINER.Caption.N_CTX ${NCTX} \
    TRAINER.Caption.CSC ${CSC} \
    TRAINER.Caption.CLASS_TOKEN_POSITION ${CTP} \
    TRAINER.Caption.use_evidence True\
    DATASET.dataset_select B \
    TRAIN.ema False \
    TEST.multi_model "('best', 'difft',)" \
    TEST.multi_scale "(2,3,4,5)" \
    TEST.use_freq True \
    TEST.save_pth True \
    TEST.save_name ./train_output_b_more_patchs_5/data_evidence.pth \
    DATALOADER.TEST.BATCH_SIZE 8
done

##
for SEED in 1 # 2 3
do
    DIR=./train_output_b_more_patchs_5/evaluation
    echo "Run this job andsave the output to ${DIR}"
    python -m torch.distributed.launch --nproc_per_node=1 --master_port=${port} --use_env train_caption.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir ./best_model \
    --eval-only \
    TRAINER.Caption.N_CTX ${NCTX} \
    TRAINER.Caption.CSC ${CSC} \
    TRAINER.Caption.CLASS_TOKEN_POSITION ${CTP} \
    DATASET.dataset_select B \
    TRAIN.ema False \
    TEST.multi_model "('zema','diff','diffh',)" \
    TEST.multi_scale "(2,3,4,5)" \
    TEST.save_pth True \
    TEST.save_name ./train_output_b_more_patchs_5/data.pth \
    DATALOADER.TEST.BATCH_SIZE 8
done

## ema模型训练prompt长度是64，单独分开测
for SEED in 1 # 2 3
do
    DIR=./train_output_b_more_patchs_5/evaluation
    echo "Run this job andsave the output to ${DIR}"
    python -m torch.distributed.launch --nproc_per_node=1 --master_port=${port} --use_env train_caption.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir ./best_model \
    --eval-only \
    TRAINER.Caption.N_CTX 64 \
    TRAINER.Caption.CSC ${CSC} \
    TRAINER.Caption.CLASS_TOKEN_POSITION ${CTP} \
    DATASET.dataset_select B \
    TRAIN.ema False \
    TEST.multi_model "('ema',)" \
    TEST.multi_scale "(2,3,4,5)" \
    TEST.save_pth True \
    TEST.save_name ./train_output_b_more_patchs_5/data_ema.pth \
    DATALOADER.TEST.BATCH_SIZE 8
done

#output/${run_ID}/${TRAINER}/${CFG}/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}\