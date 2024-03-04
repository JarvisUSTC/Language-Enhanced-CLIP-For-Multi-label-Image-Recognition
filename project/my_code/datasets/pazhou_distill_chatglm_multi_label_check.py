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
import jsonlines
import nltk
# nltk.download('punkt')
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

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


soft_hard_cls = ['bicycle', 'truck', 'bench', 'suitcase', 'frisbee', 'snowboard', 'bottle', 'cup', 'fork', 'bowl', 'apple', 'sandwich', 'orange', 'carrot', 'chair', 'dining table', 'mouse', 'keyboard', 'cell phone', 'refrigerator', 'book', 'vase']
hard_cls = ['parking meter', 'backpack', 'handbag', 'knife', 'spoon', 'potted plant', 'remote', 'microwave', 'toaster', 'scissors', 'hair drier', 'toothbrush']
total_hard_cls = hard_cls + soft_hard_cls
challenge_hard_cls = ['parking meter', 'backpack', 'handbag', 'knife', 'spoon', 'remote', 'toaster', 'scissors', 'hair drier']

soft_hard_index = [clsname2idx_[cname] for cname in soft_hard_cls]
hard_index = [clsname2idx_[cname] for cname in hard_cls]
total_hard_index = [clsname2idx_[cname] for cname in total_hard_cls]
challenge_hard_index = [clsname2idx_[cname] for cname in challenge_hard_cls]



def random_sampling(lst, M):
    # 检查列表长度是否小于3，如果是，则不进行采样
    if len(lst) < 2:
        return [lst]

    samples = []
    L = len(lst)

    for _ in range(M):
        # 随机生成采样个数 N，范围在2到L之间
        N = random.randint(2, L)
        
        # 随机选择 N 个元素进行采样
        sampled_elements = random.sample(lst, N)
        
        samples.append(sampled_elements)

    return samples

def intersec_list(listA,listB):
    return list(set(listA).intersection(set(listB)))

@DATASET_REGISTRY.register()
class pazhou_distill_chatglm_multi_label_check(DatasetBase):
    def __init__(self, cfg):
        
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        root = os.path.join(root, f"official_{cfg.DATASET.dataset_select.lower()}/")
        caption_feat_root = os.path.abspath(os.path.expanduser(cfg.DATASET.caption_feat_root))
        if cfg.TRAIN.hard_data == 'total':
            hard_list = total_hard_index
            hard_cls_list = total_hard_cls
        elif cfg.TRAIN.hard_data == 'hard':
            hard_list = hard_index
            hard_cls_list = hard_cls
        elif cfg.TRAIN.hard_data == 'soft':
            hard_list = soft_hard_index
            hard_cls_list = soft_hard_cls


        with open(join(root, 'classes.txt'), 'r') as f:
            object_categories = f.readlines()
        object_categories = [i.strip() for i in object_categories]
        cls_num = len(object_categories)

        self.dataset_dir = os.path.join(root, 'images/')

        with open(join(root, f'imnames_final{cfg.DATASET.dataset_select}.json'), 'r') as f:
            imnames_a = json.load(f)

        test = []
        for idx, imgid in enumerate(imnames_a):
            tmp_label = torch.zeros(cls_num)
            item_ = Datum(impath=join(self.dataset_dir, imgid.split('/')[-1]), label=tmp_label, classname='')
            test.append(item_)

        # ===================  training captions
        caption_feat_root = join(caption_feat_root, 'generated_captions/')
        challenge_caption_feat_root = join(caption_feat_root, 'challenge/')
            
        # if exists, just read label
        if os.path.exists(join(caption_feat_root, f'{cfg.TRAIN.Caption_name}_labels.pkl')):
            with open(join(caption_feat_root, f'{cfg.TRAIN.Caption_name}_labels.pkl'), 'rb') as f:
                word_based_caption = pickle.load(f)
            sample_capid = word_based_caption.keys()
        else:
            with open(join(caption_feat_root, f'{cfg.TRAIN.Caption_name}.json'), 'r') as f:
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
            with open(join(caption_feat_root, f'{cfg.TRAIN.Caption_name}_labels.pkl'), 'wb') as f:
                pickle.dump(word_based_caption, f)
            with open(join(caption_feat_root, f'{cfg.TRAIN.Caption_name}_filterword_empty.pkl'), 'wb') as f:
                pickle.dump(capid_empty_filter, f)

        if os.path.exists(join(caption_feat_root, f'{cfg.TRAIN.Caption_name}_all_caption_tokenized.pkl')):
            with open(join(caption_feat_root, f'{cfg.TRAIN.Caption_name}_all_caption_tokenized.pkl'), 'rb') as f:
                prompts = pickle.load(f)
        else:
            prompts = torch.cat([clip.tokenize(anno_id2path[p]['caption']) for p in sample_capid])
            with open(join(caption_feat_root, f'{cfg.TRAIN.Caption_name}_all_caption_tokenized.pkl'), 'wb') as f:
                pickle.dump(prompts, f)
        
        sample_capid_inverse_idx = {}
        for i, j in enumerate(sample_capid):
            sample_capid_inverse_idx[j] = i

        # =================================================
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
        
        def contain_chinese(check_str):
            for ch in check_str:
                if '\u4e00' <= ch <= '\u9fa5':
                    return True
            return False
        # =============================================================
        train = []
        # =============================================================

        # #######################################################################
        # ## Caption1: single label 
        # #######################################################################

        single_label_list = ['ChatGLM_single_label_1.json','ChatGLM_single_label_2.json','ChatGLM_single_label_3.json','ChatGLM_single_label_4.json', \
                             'ChatGLM_single_label_5.json']
        for slabel in single_label_list:
            with open(join(caption_feat_root, slabel), 'r') as f:
                single_caption_info = json.load(f)
            
            for cidx,c in single_caption_info.items():
                if int(cidx) not in hard_list: continue
                for i in c:
                    if not contain_chinese(i) and len(i)>5 and i[0].isdigit() and len(i) < 150:
                    
                        L = get_class(i)
                        # cidx = clsname2idx_[cname]
                        L[int(cidx)] = 1
                        text = clip.tokenize(' '.join(i.split('. ')[1:]))
                        # print("text shape:",text.shape)
                        train.append((text.squeeze(0), torch.tensor(L)))

        #######################################################################
        ## Caption1: challenge_data 
        #######################################################################
        if cfg.TRAIN.challenge_data:
            count = 1
            challenge_label_list = os.listdir(challenge_caption_feat_root)
            for slabel in tqdm(challenge_label_list,total=len(challenge_label_list)):
                with open(join(challenge_caption_feat_root, slabel), 'r') as f:
                    for line in jsonlines.Reader(f):
                        labels = line['labels']
                        multi_label = [0] * cls_num
                        for cname in labels:
                            multi_label[clsname2idx_[cname]] = 1
                        captions = line['captions']
                        for i in captions:
                            if len(i.split('. ')) > 1 and not contain_chinese(i) and len(i)>5 and i[0].isdigit() and len(i) < 150:
                            # cidx = clsname2idx_[cname]
                                text = clip.tokenize(' '.join(i.split('. ')[1:]))
                                # print("text shape:",text.shape)
                                train.append((text.squeeze(0), torch.tensor(multi_label)))
                                count+=1
            print("===== challenge Data: {} nums of word filtered caption  =====".format(count))
        # =============================================================

        # =============================================================
        num_filter_hard = 0
        for capid in word_based_caption:
           
            gt = torch.tensor(word_based_caption[capid])
            assert len(word_based_caption[capid])>0, f' len is {len(word_based_caption[capid])}'
            indices = torch.nonzero(gt == 1)[0].tolist()
            if not isinstance(indices,list): 
                print(f'{capid}:{word_based_caption[capid]}:{indices}')
                continue
            ious = intersec_list(hard_list, indices)
            if len(ious)>0:
                i = sample_capid_inverse_idx[capid]
                multi_label = [0] * cls_num
                for index in ious: multi_label[index] = 1
                item_ = (prompts[i], torch.tensor(multi_label))
                train.append(item_)
                num_filter_hard +=1
        print("===== Caption Distill total Data: {} nums of word filtered caption  =====".format(len(word_based_caption)))
        print("===== Caption Distill filterd Data: {} nums of word filtered caption  =====".format(num_filter_hard))

        # default template
        default_prompt_num = 10 # 1  100
        for i in range(cls_num):
            
            label = [0] * cls_num
            if i in hard_list:
                label[i] = 1
                tmp_p = clip.tokenize(prompt_template.format(object_categories[i]))[0]
                for j_ in range(default_prompt_num-1):
                    train.append((tmp_p, torch.tensor(label)))
            
                for cur_temp in IMAGENET_TEMPLATES:
                    tmp_p = clip.tokenize(cur_temp.format(object_categories[i]))[0]
                    train.append((tmp_p, torch.tensor(label)))

            if i == 0:
                for j in range(i + 1, cls_num):
                    if j in hard_list:
                        tmp_p = clip.tokenize(prompt_template.format(f"{object_categories[i]} and a {object_categories[j]}"))[0]
                        multi_label = [0] * cls_num
                        multi_label[i] = 1
                        multi_label[j] = 1
                        for j_ in range(default_prompt_num-1):
                            train.append((tmp_p, torch.tensor(multi_label)))

                        for cur_temp in IMAGENET_TEMPLATES:
                            tmp_p = clip.tokenize(cur_temp.format(object_categories[i]))[0]
                            train.append((tmp_p, torch.tensor(multi_label)))

                            # #进一步增强挑战性难样本
                            # if j in challenge_hard_index:
                            #     for j_ in range(default_prompt_num-1):
                            #         train.append((tmp_p, torch.tensor(multi_label)))

            
    
        # =====================================================================
        with open(join(caption_feat_root,"category_sets.txt"),'r')  as f:
            category_sets = f.readlines()

        for idx, cate_text in enumerate(category_sets):
            cname_list = cate_text.strip('\n').split(',')
            ious = intersec_list(hard_cls_list, cname_list)
            if len(ious)>0:
                text = prompt_template.format(f' and a '.join(ious))
            
                multi_label = [0] * cls_num
                if len(ious)>0:
                    for cname in ious:
                        multi_label[clsname2idx_[cname]] = 1
                    
                tmp_p = clip.tokenize(text)[0]
                train.append((tmp_p, torch.tensor(multi_label)))
                

                # if len(ious)>0: 
                #     for cur_temp in IMAGENET_TEMPLATES:
                #         tmp_p = clip.tokenize(cur_temp.format(f' and a '.join(ious)))[0]
                #         train.append((tmp_p, torch.tensor(multi_label)))
                #         ct+=1

                #######################################################################
                ## 针对通用难样本的随机采样策略
                #######################################################################
                # M = 3
                # if len(ious)>0: # and 'toaster' in ious:
                #     result = random_sampling(ious, M)
                #     for sample in result:
                #         for cname in sample: multi_label[clsname2idx_[cname]] = 1
                #         text = prompt_template.format(f' and a '.join(sample))
                #         tmp_p = clip.tokenize(text)[0]
                #         train.append((tmp_p, torch.tensor(multi_label)))
                #         ge_count+=1

                
                #######################################################################
                ## 增强分数很低的小样本类别
                #######################################################################
                
                # if len(ious)>0: # and 'toaster' in ious:
                #     challeng_ious = intersec_list(ious, challenge_hard_cls)
                    
                #     if len(challeng_ious) > 0:
                        
                #         text = prompt_template.format(f' and a '.join(challeng_ious))

                #         multi_label = [0] * cls_num
                #         for cname in challeng_ious:
                #             multi_label[clsname2idx_[cname]] = 1
                            
                #         tmp_p = clip.tokenize(text)[0]
                #         train.append((tmp_p, torch.tensor(multi_label)))
                #         ch_count+=1
        # =====================================================================


        if not os.path.exists(join(caption_feat_root, f'{cfg.TRAIN.Caption_name}_class_freq.pkl')):
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
