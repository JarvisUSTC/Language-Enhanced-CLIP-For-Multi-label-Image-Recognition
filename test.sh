cd project/my_code/
cd ./Dassl.pytorch-master/
python setup.py develop

cd ..

## inference
bash run_eval.sh pazhou_distill_chatglm_multi_label_mix rn50-1k-ChatGLM_multi_labels_mix end 16 False eval 0 25800

python gen_final_ans.py
