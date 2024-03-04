import json
import os
from transformers import AutoTokenizer, AutoModel
import os
from tqdm import tqdm
import argparse
from data_helpers import coco_object_categories



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

def read_json(path):
    with open(path,'r') as f:
        c = json.load(f)
    return c

def gen_captions_single(args):

    compositions_info_path = args.compositions_info_path
    save_root = args.save_root
    os.makedirs(save_root,exist_ok=True)
    st=args.st
    ed=args.ed
    loop_num = args.loop_num
    
    #option:
    if args.glm_offline:
        model, tokenizer = get_glm(model_dir=args.model_dir)
    else:
        model, tokenizer = get_glm()
    
    num = 10
    for t in range(loop_num):
        save_path = os.path.join(save_root, f'ChatGLM_single_label_{t+1}.json')
        caps_dict = {}
        for idx,names in tqdm(enumerate(coco_object_categories),total=len(coco_object_categories)):

            TEMPLATE_FOR_JUDGE_CAPTION = [f"Determine if you can provide some sentences describing a scene where {names} appear together. If so, answer 'True', otherwise answer 'False'. Explain the reason simply.", 
                                    f"Suppose you are an image describer and I want you to help us to describe various images in a real scene that contain some specific category of objects according to that category. please generate 10 very simple sentences that are distinguishable, concise and realistic. These sentences describe 10 different images where {names} MUST appear.",
                                    f"Suppose you are a scene describer and I want you to help us to describe various scenes in real life that contain a specific category of objects according to that category.  For the category:{names}, please generate {num} descriptive sentences that are as different as possible, require brevity, show the characteristics of the category, and are distinguishable.",
                                    "Based on the previous considerations, determine if this sentence '{caption}' is a good description of a scene where {names} appear together. If so, answer 'True', otherwise answer 'False'. Explain the reason simply."]
            
        
            response_judge, history = model.chat(tokenizer, TEMPLATE_FOR_JUDGE_CAPTION[1], history=[], max_length=1200, top_p=0.95)
            # print(response_judge)
            captions = response_judge.split('\n')
            caps_dict[idx] = captions
            print(caps_dict[idx])
            with open(save_path,'w',encoding='utf-8') as f:
                json.dump(caps_dict,f,indent=4)
            f.close()
        
       
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--glm_offline", action="store_true", help="use offline model of chatglm")
    parser.add_argument("--compositions_info_path", type=str, default="./compositions_of_image.json", help="path to category compositions info")
    parser.add_argument("--model_dir", type=str, default="/root/autodl-tmp/chatglm-6b/", help="offline chatglm-6b directory")
    parser.add_argument("--save_root", type=str, default="./gen_caption/", help="offline chatglm-6b directory")
    parser.add_argument("--st", type=int, default=0, help="start")
    parser.add_argument("--ed", type=int, default=4, help="end")
    parser.add_argument("--loop_num", type=int, default=4, help="end")
    
    args = parser.parse_args()
    
    print(args)
    gen_captions_single(args)
