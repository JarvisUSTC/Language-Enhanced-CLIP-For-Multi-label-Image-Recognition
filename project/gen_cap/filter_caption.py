import json
import os
from transformers import AutoTokenizer, AutoModel
import os
from tqdm import tqdm
import argparse

import tqdm
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

from data_helpers import *

object_categories = coco_object_categories
classname_synonyms = coco_classname_synonyms
clsname2idx_ = {}
nameset_compound = set()
nameset = set()
for idx, synset in enumerate(classname_synonyms):
    for n in synset:
        clsname2idx_[n] = idx

        if ' ' in n:
            nameset_compound.add(n)
            m = n.replace(' ', '')
            clsname2idx_[m] = idx
            nameset.add(m)
        else:
            nameset.add(n)

wnl = WordNetLemmatizer()
def get_class(caption):
    def get_wordnet_pos(tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return None

    cls_num = 80
    cap = caption.lower()
    noum_list = word_tokenize(cap)
    tagged_sent = pos_tag(noum_list) 

    lemmas_sent = []
    for tag in tagged_sent:
        wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
        lemmas_sent.append(wnl.lemmatize(tag[0], pos=wordnet_pos))

    cap = ' ' + ' '.join(lemmas_sent) + ' '

    labels = []
    L = [0] * cls_num
    flag = 0
    for name in nameset_compound:
        name_ = ' ' + name + ' '
        if (name_ in cap):
            L[clsname2idx_[name]] = 1
            labels.append(name)
            flag = 1
            cap = cap.replace(name_, ' ')
    for name in nameset:
        name_ = ' ' + name + ' '
        if (name_ in cap):
            L[clsname2idx_[name]] = 1
            labels.append(name)
            flag = 1
            cap = cap.replace(name_, ' ')

    return labels

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

def simple_filter_cap(args):
    
    cap_dir = args.caption_dir
    import re
    gen_caption_list = os.listdir(cap_dir)
    slist_all = []
    for fname in gen_caption_list:
        slist = []
        f = open(os.path.join(cap_dir,fname),'r')
        #### 处理txt
        if fname.endswith('.txt'):
            # pass
            info_list = [line.strip('\n') for line in f.readlines()]
            for sentence in info_list:
                match = re.findall(r'\d+\.\s', sentence)
                if len(match) ==1:
                    processed_sentence = sentence.split(match[0])[-1]
                    slist.append(processed_sentence)
                # 多个句子合并情况 
                elif len(match) >1:
                    # print(f'{fname}:{sentence}')
                    sentence = sentence.split(match[0])[-1]
                    for idx,m in enumerate(match[1:]):
                        processed_sentence = sentence.split(m)[0]
                        sentence = sentence.split(m)[-1]
                        slist.append(processed_sentence)
                    slist.append(sentence.split(m)[-1])


        elif fname.endswith('.json'):
            cap_dict = json.load(f)
            for k,value in cap_dict.items():
            # value:list
                for idx, sentence in enumerate(value):
                    match = re.findall(r'\d+\.\s', sentence)
                    if len(match) > 0:
                        if len(match)>1:
                            # Example:"A airplane xxxxx at 11:59. A person xxxxx"
                            if ':' in sentence:
                                print(f'{fname}:{sentence}')
                                processed_sentence = sentence.split(match[0])[-1]
                                slist.append(processed_sentence)
                            # 多个句子合并情况
                            else:
                                sentence = sentence.split(match[0])[-1]
                                for idx,m in enumerate(match[1:]):
                                    processed_sentence = sentence.split(m)[0]
                                    sentence = sentence.split(m)[-1]
                                    slist.append(processed_sentence)
                                slist.append(sentence.split(m)[-1])
                        # 只有开始位置 ‘数字. '
                        else:
                            processed_sentence = sentence.split(match[0])[-1]
                            slist.append(processed_sentence)
                            
        def contains_chinese(s):  
            if re.search(r'[\u4e00-\u9fff]', s):  
                return True  
            else:  
                return False

        filtered_captions = []

        for caption in slist:
            if contains_chinese(caption):
                continue
            filtered_captions.append(caption)
        
        slist_all.extend(filtered_captions)

    return slist_all

def filter_cap(args):
    
    filtered_captions = simple_filter_cap(args)

    if args.glm_offline:
        model, tokenizer = get_glm(model_dir=args.model_dir)
    else:
        model, tokenizer = get_glm()
        
    save_root = args.save_root

    system_prompt = "Suppose you are an image describer and I want you to help us to determine if the provided caption is a good description (need to be distinguishable, concise and realistic) of a real life scene. Try to keep sentences with verbs that indicate interaction between objects, such as hit, wear, play. If so, answer 'True', otherwise answer 'False'. For example, 'A kitchen with a microwave, refrigerator, and sink.' is 'True'. 'Gorilla waving on the moon' is 'False'."
    user_prompt = """
    Provided caption: {caption}.
    """
    selected_captions_for_training = []
    
    for caption in tqdm(filtered_captions):
        response_judge, history = model.chat(tokenizer, system_prompt + user_prompt.format(caption=caption), history=[], max_length=1200, top_p=0.95, temperature=0.3)
        if 'True' in response_judge and 'False' not in response_judge:
            selected_captions_for_training.append(caption)

        if len(selected_captions_for_training) % 5000 == 0:
            # save
            selected_captions_for_training_with_label = []
            selected_captions_for_training_with_label_json = []
            for i in tqdm(selected_captions_for_training,total=len(selected_captions_for_training)):
                if len(i)>5 and len(i) < 150:
                    multi_labels = get_class(i)
                    selected_captions_for_training_with_label.append((i, multi_labels))
                    selected_captions_for_training_with_label_json.append({
                        'id': len(selected_captions_for_training_with_label) - 1,
                        'caption': i,
                        'labels': multi_labels
                    })
            json.dump(selected_captions_for_training_with_label_json, open(f'{save_root}/ChatGLM_multi_labels_filtered.json', 'w'), indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--glm_offline", action="store_true", help="use offline model of chatglm")
    parser.add_argument("--model_dir", type=str, default="/root/autodl-tmp/chatglm-6b/", help="offline chatglm-6b directory")
    parser.add_argument("--save_root", type=str, default="./", help="save filtered caption directory")
    parser.add_argument("--caption_dir", type=str, default="./gen_caption/", help="generated caption directory")
    
    args = parser.parse_args()
    
    print(args)
    filter_cap(args)