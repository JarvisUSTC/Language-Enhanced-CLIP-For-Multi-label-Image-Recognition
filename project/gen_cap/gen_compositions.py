import json
import os
from transformers import AutoTokenizer, AutoModel
import os
from tqdm import tqdm
import argparse
import random

from data_helpers import *

object_categories = coco_object_categories

def get_glm(model_dir=None):
    if model_dir is not None:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, revision="v1.1.0")

        # 请参考 THUDM/chatglm-6b 官方 GitHub 仓库 (https://github.com/THUDM/ChatGLM-6B) 中对模型运行所需资源的介绍
        # 本样例在配置为 2080ti(11GB) * 2 的开发机上运行，因此设置 num_gpus=2
        model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).half().cuda()
    else:
        tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True, revision="v1.1.0")
        # 请参考 THUDM/chatglm-6b 官方 GitHub 仓库 (https://github.com/THUDM/ChatGLM-6B) 中对模型运行所需资源的介绍
        # 本样例在配置为 2080ti(11GB) * 2 的开发机上运行，因此设置 num_gpus=2
        model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().cuda()
    model= model.eval()
    return model, tokenizer

def gen_compositions(args):

    loop_num = args.loop_num
    
    #option:
    if args.glm_offline:
        model, tokenizer = get_glm(model_dir=args.model_dir)
    else:
        model, tokenizer = get_glm()
        
    save_root = args.save_root
    
    compositions = {}
    id = 0
    for t in tqdm(range(loop_num)):
        temp = random.sample(object_categories, 10)
        TEMPLATE_FOR_GENERATE_OBJECT_COMPOSITIONS = f"There are several categories of objects here [{', '.join(temp)}]. Imagine some categories of objects may appear in a natural and real-life scene at the same time, such as human and car appearing in the road, while elephant and airplane do not appear in the same scene. Please output no more than 5 category combinations that You are very Certain that they must appear in a natural and real-life scene at the same time."

        response_judge, history = model.chat(tokenizer, TEMPLATE_FOR_GENERATE_OBJECT_COMPOSITIONS, history=[], max_length=1200, top_p=0.95, temperature=0.3)
        all_sentences = response_judge.lower().split('\n')
        for s in all_sentences:
            composition = []
            for c in temp:
                if c in s:
                    composition.append(c)
            if len(composition) > 1:
                compositions[id] = {'category_name': composition}
                id += 1
                
    json.dump(compositions, open(os.path.join(save_root, 'compositions_of_image.json'), 'w'), indent=4)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--glm_offline", action="store_true", help="use offline model of chatglm")
    parser.add_argument("--model_dir", type=str, default="/root/autodl-tmp/chatglm-6b/", help="offline chatglm-6b directory")
    parser.add_argument("--save_root", type=str, default="./", help="offline chatglm-6b directory")
    parser.add_argument("--loop_num", type=int, default=4, help="end")
    
    args = parser.parse_args()
    
    print(args)
    gen_compositions(args)