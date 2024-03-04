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

@DATASET_REGISTRY.register()
class pazhou_distill_chatglm(DatasetBase):
    def __init__(self, cfg):
        
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        root = os.path.join(root, "A_datasets/")

        with open(join(root, 'classes.txt'), 'r') as f:
            object_categories = f.readlines()
        object_categories = [i.strip() for i in object_categories]
        cls_num = len(object_categories)

        self.dataset_dir = os.path.join(root, 'dataset_A')

        with open(join(root, 'imnames_A.json'), 'r') as f:
            imnames_a = json.load(f)

        test = []
        for idx, imgid in enumerate(imnames_a):
            tmp_label = torch.zeros(cls_num)
            item_ = Datum(impath=join(root, 'dataset_A', imgid.split('/')[-1]), label=tmp_label, classname='')
            test.append(item_)

        # ===================  training captions
        caption_feat_root = os.getcwd()
        with open(join(caption_feat_root, f'generated_captions/{cfg.TRAIN.Caption_name}.json'), 'r') as f:
            texts_dict = json.load(f)
        
        all_prompts = []
        all_labels = []
        for cls_idx in tqdm(range(cls_num), desc="Tokenizing ChatGLM texts"):
            cls_texts = texts_dict[str(cls_idx)]
            curcls_prompts = torch.cat([clip.tokenize(p, truncate=True) for p in cls_texts])  # torch.cat([clip.tokenize(p) for p in prompts])

            all_prompts.append(curcls_prompts)

            cls_labels = torch.tensor([[0] * cls_num] * len(cls_texts)) # torch.zeros((len(cls_texts), cls_num), dtype=torch.float32)
            cls_labels[:, cls_idx] = 1
            all_labels.append(cls_labels)
            
        all_prompts = torch.cat(all_prompts)
        all_labels = torch.cat(all_labels)
        
        print('===== chatglm generate {} sentences ====='.format(all_prompts.shape[0]), all_prompts.shape, all_labels.shape)
        
        # #
        train = []
        if not cfg.TRAIN.IF_ablation:
            for i in range(all_prompts.shape[0]):
                item_ = (all_prompts[i], all_labels[i])
                train.append(item_)
        print("===== Caption Distill Data: {} nums of word filtered caption  =====".format(len(train)))


        super().__init__(train_x=train, val=test[0::100], test=test, \
            num_classes=len(object_categories), classnames=object_categories, \
            lab2cname={idx: classname for idx, classname in enumerate(object_categories)})
