import os
from os.path import join
from re import L
import pickle5 as pickle
import random
from scipy.io import loadmat
from collections import defaultdict
import torch
import json
from tqdm import tqdm
from clip import clip
from clip.model import convert_weights

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import read_json, mkdir_if_missing
from clip import tokenize

from .data_helpers import *

from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import re
from tqdm import tqdm
import random

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

def contain_chinese(check_str):
    for ch in check_str:
        if '\u4e00' <= ch <= '\u9fa5':
            return True
    return False

def random_sampling(lst, M):
    # 检查列表长度是否小于3，如果是，则不进行采样
    if len(lst) < 4:
        return [lst]

    samples = []
    L = len(lst)

    for _ in range(M):
        # 随机生成采样个数 N，范围在3到L之间
        N = random.randint(3, L)
        
        # 随机选择 N 个元素进行采样
        sampled_elements = random.sample(lst, N)
        
        samples.append(sampled_elements)

    return samples

@DATASET_REGISTRY.register()
class pazhou_distill_chatglm_multi_label_zuan(DatasetBase):
    def __init__(self, cfg):
        
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        root = os.path.join(root, f"official_{cfg.DATASET.dataset_select.lower()}/")
        caption_feat_root = os.path.abspath(os.path.expanduser(cfg.DATASET.caption_feat_root))

        with open(join(root, 'classes.txt'), 'r') as f:
            object_categories = f.readlines()
        object_categories = [i.strip() for i in object_categories]
        cls_num = len(object_categories)

        self.dataset_dir = os.path.join(root, f'dataset_{cfg.DATASET.dataset_select}')

        with open(join(root, f'imnames_{cfg.DATASET.dataset_select}.json'), 'r') as f:
            imnames_a = json.load(f)

        test = []
        for idx, imgid in enumerate(imnames_a):
            tmp_label = torch.zeros(cls_num)
            item_ = Datum(impath=join(root, f'dataset_{cfg.DATASET.dataset_select}', imgid.split('/')[-1]), label=tmp_label, classname='')
            test.append(item_)

        # ===================  training captions
        caption_feat_root = join(caption_feat_root, 'generated_captions/')
        
        train = []
        #######################################################################
        ## Caption1: single label 
        #######################################################################
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

            L = [0] * cls_num
            flag = 0
            for name in nameset_compound:
                name_ = ' ' + name + ' '
                if (name_ in cap):
                    L[clsname2idx_[name]] = 1
                    flag = 1
                    cap = cap.replace(name_, ' ')
            for name in nameset:
                name_ = ' ' + name + ' '
                if (name_ in cap):
                    L[clsname2idx_[name]] = 1
                    flag = 1
                    cap = cap.replace(name_, ' ')

            return L
        
        single_label_list = ['ChatGLM_single_label_1.json','ChatGLM_single_label_2.json','ChatGLM_single_label_3.json','ChatGLM_single_label_4.json', \
                             'ChatGLM_single_label_5.json']
        for slabel in single_label_list:
            with open(join(caption_feat_root, slabel), 'r') as f:
                single_caption_info = json.load(f)
            
            for cidx,c in single_caption_info.items():
                for i in c:
                    if not contain_chinese(i) and len(i)>5 and i[0].isdigit() and len(i) < 150:
                    
                        L = get_class(i)
                        # cidx = clsname2idx_[cname]
                        L[int(cidx)] = 1
                        text = clip.tokenize(' '.join(i.split('. ')[1:]))
                        # print("text shape:",text.shape)
                        train.append((text.squeeze(0), torch.tensor(L)))
            
        #######################################################################
        ## Caption1: multi label 2k v2 &v3
        #######################################################################
        caption_list = f'{cfg.TRAIN.Caption_name}'.split(' ')
        for caption_name in caption_list:  
            
            if os.path.exists(join(caption_feat_root, f'{caption_name}_labels.pkl')):
                with open(join(caption_feat_root, f'{caption_name}_labels.pkl'), 'rb') as f:
                    word_based_caption = pickle.load(f)
                sample_capid = word_based_caption.keys()
                # print(sample_capid)
            else:
                with open(join(caption_feat_root, f'{caption_name}.json'), 'r') as f:
                    caption_info = json.load(f)

                anno_id2path = {}
                for i in caption_info:
                    anno_id2path[i["id"]] = i
                sample_capid = list(anno_id2path.keys())
                # print(i.keys())
                print("captions_train2017 nums:", len(anno_id2path))

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

                word_based_caption = {} # capid 2 cls labels
                capid_empty_filter = set()
                wnl = WordNetLemmatizer()
                for i, capid in enumerate(tqdm(sample_capid)):
                    cap = anno_id2path[capid]['caption'].lower()
                    noum_list = word_tokenize(cap)
                    tagged_sent = pos_tag(noum_list) 
                    # print(tagged_sent)
                    # break

                    lemmas_sent = []
                    for tag in tagged_sent:
                        wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
                        lemmas_sent.append(wnl.lemmatize(tag[0], pos=wordnet_pos))
                    # print(lemmas_sent)

                    cap = ' ' + ' '.join(lemmas_sent) + ' '

                    L = [0] * cls_num
                    flag = 0
                    for name in nameset_compound:
                        name_ = ' ' + name + ' '
                        if (name_ in cap):
                            L[clsname2idx_[name]] = 1
                            flag = 1
                            cap = cap.replace(name_, ' ')
                    for name in nameset:
                        name_ = ' ' + name + ' '
                        if (name_ in cap):
                            L[clsname2idx_[name]] = 1
                            flag = 1
                            cap = cap.replace(name_, ' ')

                    if flag:
                        word_based_caption[capid] = L
                    else:
                        capid_empty_filter.add(capid)
                sample_capid = word_based_caption.keys()

                print('===== Filtered by words all captions, num of captions contains object {}, num of caption empty {} ====='.format(len(word_based_caption), len(capid_empty_filter)))
                with open(join(caption_feat_root, f'{caption_name}_labels.pkl'), 'wb') as f:
                    pickle.dump(word_based_caption, f)
                with open(join(caption_feat_root, f'{caption_name}_filterword_empty.pkl'), 'wb') as f:
                    pickle.dump(capid_empty_filter, f)

            if os.path.exists(join(caption_feat_root, f'{caption_name}_all_caption_tokenized.pkl')):
                with open(join(caption_feat_root, f'{caption_name}_all_caption_tokenized.pkl'), 'rb') as f:
                    prompts = pickle.load(f)
            else:
                prompts = torch.cat([clip.tokenize(anno_id2path[p]['caption']) for p in sample_capid])
                with open(join(caption_feat_root, f'{caption_name}_all_caption_tokenized.pkl'), 'wb') as f:
                    pickle.dump(prompts, f)

            sample_capid_inverse_idx = {}
            for i, j in enumerate(sample_capid):
                sample_capid_inverse_idx[j] = i
        

            for capid in word_based_caption:
                i = sample_capid_inverse_idx[capid]
                # print("prompt shape:",prompts[i].shape)
                item_ = (prompts[i], torch.tensor(word_based_caption[capid]))
                train.append(item_)
            print("===== Caption Distill Data: {} nums of word filtered caption  =====".format(len(word_based_caption)))

        # default template: a photo of 
        #######################################################################
        ## Default template: Single Label & 暴力组合两个类别
        #######################################################################
        default_prompt_num = 10 # 1  100
        for i in range(cls_num):
            label = [0] * cls_num
            label[i] = 1
            # print("get temp prompt:", prompt_template.format(object_categories[i]))
            tmp_p = clip.tokenize(prompt_template.format(object_categories[i]))[0]
            for j_ in range(default_prompt_num-1):
                train.append((tmp_p, torch.tensor(label)))
            
            for cur_temp in IMAGENET_TEMPLATES:
                tmp_p = clip.tokenize(cur_temp.format(object_categories[i]))[0]
                train.append((tmp_p, torch.tensor(label)))

            # if i == 0:
            for j in range(i + 1, cls_num):
                tmp_p = clip.tokenize(prompt_template.format(f"{object_categories[i]} and a {object_categories[j]}"))[0]
                multi_label = [0] * cls_num
                multi_label[i] = 1
                multi_label[j] = 1
                for j_ in range(default_prompt_num-1):
                    train.append((tmp_p, torch.tensor(multi_label)))
        
        if cfg.TRAIN.add_few_shot:
            # default template for few-shot classes
            with open(join(caption_feat_root,'components_of_few_shot_classes.json'),'r') as f:
                few_shot_info = json.load(f)
            
            for key, value in few_shot_info.items():
                # key is the class name, value is the list of related classes (do not appear in the provided class names)
                for cname in value:
                    tmp_p = clip.tokenize(prompt_template.format(f"{key} and a {cname}"))[0]
                    multi_label = [0] * cls_num
                    multi_label[clsname2idx_[key]] = 1
                    train.append((tmp_p, torch.tensor(multi_label)))

        
        #######################################################################
        ## Default template: read categories 
        ## 全类别组合
        #######################################################################
        # with open(join(caption_feat_root,"category_sets.txt"),'r')  as f:
        #     category_sets = f.readlines()
        
        # all_cates = set()
        # for idx, cate_text in enumerate(category_sets):
        #     cname_list = cate_text.strip('\n').split(',')
       
        #     cate_name = tuple(sorted(tuple(set(cname_list))))
        #     all_cates.add(cate_name)

        # cates_above2 = set()
        # for cate in all_cates:
        #     if len(cate) == 1:
        #         continue
        #     cates_above2.add(cate)

        # print(len(all_cates))
        # print(len(cates_above2))
        # cates_name_set = list(cates_above2)
        # print(f"num of categories set:{len(cates_name_set)}")
        
        # ## Full categories
        # for idx, cate_text in enumerate(category_sets):
        #     cname_list = cate_text.strip('\n').split(',')
        #     # full category set
        #     text = prompt_template.format(f' and a '.join(cname_list))
            
        #     multi_label = [0] * cls_num
        #     for cname in cname_list:
        #         multi_label[clsname2idx_[cname]] = 1
                
        #     tmp_p = clip.tokenize(text)[0]
        #     train.append((tmp_p, torch.tensor(multi_label)))
        
        
        ## Sample categories (Beta)
        # M = 5 # Sample nums
        # for idx,com in tqdm(enumerate(cates_name_set),total=len(cates_name_set)):
        #     result = random_sampling(com, M)
            
        #     multi_label = [0] * cls_num
        #     for cname in com:
        #         multi_label[clsname2idx_[cname]] = 1
                    
        #     for sample in result:
        #         # print(f"Sample {i}: {sample}")
        #         text = prompt_template.format(f' and a '.join(sample))
        #         tmp_p = clip.tokenize(text)[0]
        #         train.append((tmp_p, torch.tensor(multi_label)))
                    
        import numpy as np
        import mmcv
        gt_labels = np.stack([i[1].numpy() for i in train])
        class_freq = np.sum(gt_labels, axis=0)
        neg_class_freq = np.shape(gt_labels)[0] - class_freq
        class_freq_info = dict(gt_labels=gt_labels, class_freq=class_freq, neg_class_freq=neg_class_freq)
        mmcv.dump(class_freq_info, join(caption_feat_root, f'{cfg.TRAIN.Caption_name}_class_freq.pkl'))

        super().__init__(train_x=train, val=test[0::100], test=test, \
            num_classes=len(object_categories), classnames=object_categories, \
            lab2cname={idx: classname for idx, classname in enumerate(object_categories)})
