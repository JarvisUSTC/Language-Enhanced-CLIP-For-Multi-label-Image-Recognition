# single generation
python project/gen_cap/gen_caption_single.py --glm_offline --model_dir official_model/chatglm --save_root project/output/text_result/repro_data/ --loop_num 5

# challenge generation
python project/gen_cap/gen_caption_challenge.py --glm_offline --model_dir official_model/chatglm --save_root project/output/text_result/repro_data/challenge/ --compositions_info_path project/output/text_result/compositions_of_image.json

# Composition generation
python project/gen_cap/gen_compositions.py --glm_offline --model_dir official_model/chatglm --save_root project/output/text_result/repro_data/ --loop_num 6000
# project/gen_cap/compositions_of_image.json is our local generated compositions for competition.

# Caption generation
python project/gen_cap/gen_caption.py --glm_offline --model_dir official_model/chatglm --save_root project/output/text_result/repro_data/gen_caption --compositions_info_path project/output/text_result/compositions_of_image.json --st 0 --ed 25000 --loop_num 4
# project/output/text_result/gen_cap_local is our local generated captions for competition.

# Filter caption
python project/gen_cap/filter_caption.py --glm_offline --model_dir official_model/chatglm --caption_dir project/output/text_result/repro_data/gen_caption --save_root project/output/text_result/repro_data/
# project/data/ChatGLM_multi_labels_filtered.json is our local filtered captions for competition.