cd project/my_code/
cd ./Dassl.pytorch-master/
python setup.py develop

cd ..
## do training 

## step1. train main models  with 3 strategies
# train best evidencewoloss model
bash run_seed0.sh pazhou_distill_chatglm_multi_label_mix rn50-1k-ChatGLM_multi_labels_evidence_best15e end 16 False best 0 25800

## step2. train ema models 
# train ema model
bash run_seed1.sh pazhou_distill_chatglm_multi_label_mix rn50-1k-ChatGLM_multi_labels_ema end 64 False ema 0 25805

## step3. train zema model (mix training)
# train zema model
bash run_seed1.sh pazhou_distill_chatglm_multi_label_zema rn50-1k-ChatGLM_multi_labels_zema100e end 16 False zema 0 25806

## step4. train models with challenge data and randomly generated specific categories
# train diffh_100eEMA model
bash run_seed1.sh pazhou_distill_chatglm_multi_label_check rn50-1k-ChatGLM_multi_labels_check_diffh_100eEMA end 16 False diffh 0 25802
# train diff_100e model
bash run_seed1.sh pazhou_distill_chatglm_multi_label_check rn50-1k-ChatGLM_multi_labels_check_diff_100e end 16 False diff 0 25803
# train difft evidencewoloss model
bash run_seed1.sh pazhou_distill_chatglm_multi_label_check rn50-1k-ChatGLM_multi_labels_evidence_check_difft_15e end 16 False difft 0 25804



# get best model: move pth to best_model
mkdir -p ../best_model/best/
cp ./train_output/best/prompt_learner/model.pth.tar-5 ../best_model/best/
mv ../best_model/best/model.pth.tar-5 ../best_model/best/model.pth.tar

mkdir -p ../best_model/ema/
cp ./train_output/ema/prompt_learner/model.pth.tar-80 ../best_model/ema/
mv ../best_model/ema/model.pth.tar-80 ../best_model/ema/model.pth.tar
mkdir -p ../best_model/zema/
cp ./train_output/zema/prompt_learner/model.pth.tar-80 ../best_model/zema/
mv ../best_model/zema/model.pth.tar-80 ../best_model/zema/model.pth.tar

mkdir -p ../best_model/diffh/
cp ./train_output/diffh/prompt_learner/model.pth.tar-80 ../best_model/diffh/
mv ../best_model/diffh/model.pth.tar-80 ../best_model/diffh/model.pth.tar
mkdir -p ../best_model/diff/
cp ./train_output/diff/prompt_learner/model.pth.tar-10 ../best_model/diff/
mv ../best_model/diff/model.pth.tar-10 ../best_model/diff/model.pth.tar
mkdir -p ../best_model/difft/
cp ./train_output/difft/prompt_learner/model.pth.tar-10 ../best_model/difft/
mv ../best_model/difft/model.pth.tar-10 ../best_model/difft/model.pth.tar

