import json
import os
from transformers import AutoTokenizer, AutoModel
import os
from tqdm import tqdm
import argparse
import  jsonlines


challenge_cls = ['parking meter', 'backpack', 'handbag', 'bench','bottle','knife', 'spoon', 'chair', 'potted plant', 'mouse', 'remote', 'microwave', 'toaster', 'scissors', 'hair drier', 'toothbrush','truck']


def intersec_list(listA,listB):
    return list(set(listA).intersection(set(listB)))

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

def gen_captions(args):

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
        
    content = read_json(compositions_info_path)
    all_cates = set()
    for key, value in content.items():
        if "category_name" in value:
            cate_name = tuple(sorted(tuple(set(value["category_name"]))))
            all_cates.add(cate_name)
    cates_above2 = set()
    for cate in all_cates:
        if len(cate) == 1:
            continue
        cates_above2.add(cate)

    print(len(all_cates))
    print(len(cates_above2))

    cates_name_set = list(cates_above2)

    challenge_cates_set = []
    for idx, cates in enumerate(cates_name_set):
        ious = intersec_list(challenge_cls,cates)
        if len(ious) > 0: challenge_cates_set.append(ious)

    print("challenge cates num:",len(challenge_cates_set))

    save_dict = {}
    save_path = os.path.join(save_root, f'captions_score_challenge.jsonl')

    w = jsonlines.open(save_path, 'w')
    for idx,com in tqdm(enumerate(challenge_cates_set),total=len(challenge_cates_set)):
        names = ', '.join(com[:-1]) + ' and '+ com[-1]
        print('#'*20, '\n', com,names)
        TEMPLATE_FOR_JUDGE_CAPTION = [f"Determine if you can provide some sentences describing a scene where {names} appear together. If so, answer 'True', otherwise answer 'False'. Explain the reason simply.", 
                                  f"Suppose you are an image describer and I want you to help us to describe various images in a real scene that contain some specific category of objects according to that category. please generate 4 very simple sentences that are distinguishable, concise and realistic. These sentences describe 5 different images where {names} MUST appear together.",
                                  "Based on the previous considerations, determine if this sentence '{caption}' is a good description of a scene where {names} appear together. If so, answer 'True', otherwise answer 'False'. Explain the reason simply."]
        
        
        response_judge, history = model.chat(tokenizer, TEMPLATE_FOR_JUDGE_CAPTION[1], history=[], max_length=1200, top_p=0.95)
        captions = response_judge.split('\n')
        out_dict = {}
        out_dict['labels'] = com
        out_dict['captions'] = captions
        print(out_dict)
        w.write(out_dict)
        
       
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
    gen_captions(args)
