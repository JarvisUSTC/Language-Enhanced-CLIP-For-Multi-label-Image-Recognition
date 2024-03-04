import os.path as osp
import os
import numpy as np
import mmcv
import pdb
import copy

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from torchvision.models._utils import IntermediateLayerGetter

from torch.nn.parallel import DistributedDataParallel as DDP 
import torch.distributed as dist 

from tqdm import tqdm
import pickle5 as pickle

from .csel import SoftMarginHingeEmbeddingLoss
from .dbl import ResampleLoss

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from .utils import soft_cross_entropy, softmax_sigmoid_BCEloss, \
    norm_logits_BCEloss, sigmoid_focal_loss, sigmoid_ASL_loss, ranking_loss, ASL_loss, ranking_loss_with_cooccurrence
_tokenizer = _Tokenizer()

with open("./ChatGLM_multi_labels_filtered_22w_all_caption_text_feats.pkl", "rb") as f:
    caption_text_feats = pickle.load(f).cuda()

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    # model_path = clip._download(url, root= os.path.expanduser(f"~/.cache/clip_{dist.get_rank()}"))
    model_path = "/workspace/official_model/clip/RN50.pt"

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class BottleNeckAdapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.token_embedding = clip_model.token_embedding

    def forward(self, prompts, tokenized_prompts, if_embedding=True, if_sequence=False):
        if not if_embedding:
            tokenized_prompts = prompts
            prompts = self.token_embedding(prompts).type(self.dtype)  # [batch_size, n_ctx, d_model]
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        
        if if_sequence:
            x = x @ self.text_projection  # NLD * Dd = NLd
            return x
        else:
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            # ND * Dd = Nd
            x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
            return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model, nctx=None):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.Caption.N_CTX if nctx is None else nctx
        ctx_init = cfg.TRAINER.Caption.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init, truncate=True)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.Caption.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            
            if cfg.TRAINER.Caption.CSC:
                print("Initializing class-specific double contexts")
                ctx_vectors_double = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors_double = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors_double, std=0.02)
            
            if cfg.TRAINER.Caption.CSC:
                print("Initializing class-specific double contexts")
                # ctx_vectors_double = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
                ctx_vectors_evidence = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors_evidence = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors_evidence, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f'Initial double context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        self.ctx_double = nn.Parameter(ctx_vectors_double)  # to be optimized
        self.ctx_evidence = nn.Parameter(ctx_vectors_evidence)
        
        temperature = torch.tensor(3.0, dtype=dtype)  #  exp(3.91) = 50
        self.temperature = nn.Parameter(temperature)
        spatial_T = torch.tensor(3.0, dtype=dtype)  # 20
        self.spatial_T = nn.Parameter(spatial_T)
        ranking_scale = torch.tensor(4.0, dtype=dtype)  # 20
        self.ranking_scale = nn.Parameter(ranking_scale)

        # sigmoid_shift = torch.tensor(0.25, dtype=dtype)
        # self.sigmoid_shift = nn.Parameter(sigmoid_shift)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p, truncate=True) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS
        
        # class agnostic token suffix
        prompts_nocls = [prompt_prefix + "."] * len(classnames)
        tokenized_prompts_nocls = torch.cat([clip.tokenize(p, truncate=True) for p in prompts_nocls])
        with torch.no_grad():
            embedding_nocls = clip_model.token_embedding(tokenized_prompts_nocls).type(dtype)
        self.register_buffer("token_suffix_nocls", embedding_nocls[:, 1 + n_ctx :, :])  # EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.Caption.CLASS_TOKEN_POSITION

    def forward(self, neg_prompt_wcls=True):
        """
        Returns current learned ctx embeddings, concated with cls word embeddings.
        """
        ctx = self.ctx
        ctx_double = self.ctx_double
        ctx_evidence = self.ctx_evidence
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        if ctx_double.dim() == 2:
            ctx_double = ctx_double.unsqueeze(0).expand(self.n_cls, -1, -1)
        if ctx_evidence.dim() == 2:
            ctx_evidence = ctx_evidence.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        suffix_nocls = self.token_suffix_nocls

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )
            if neg_prompt_wcls:
                prompts_neg = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctx_double,     # (n_cls, n_ctx, dim)
                        suffix,  # (n_cls, *, dim)
                    ],
                    dim=1,
                )
                prompts_evidence = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctx_evidence,     # (n_cls, n_ctx, dim)
                        suffix,  # (n_cls, *, dim)
                    ],
                    dim=1,
                )
            else:
                prompts_neg = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctx_double,     # (n_cls, n_ctx, dim)
                        suffix_nocls,  # (n_cls, *, dim)
                    ],
                    dim=1,
                )
                prompts_evidence = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctx_evidence,     # (n_cls, n_ctx, dim)
                        suffix_nocls,  # (n_cls, *, dim)
                    ],
                    dim=1,
                )


        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts, prompts_neg, prompts_evidence, self.temperature, self.spatial_T, self.ranking_scale

class CustomCLIP(nn.Module):
    def __init__(self, args, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(args, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts # [n_cls, ctx_length]
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        
        self.model = clip_model

    def forward(self, image=None, captions=None, if_test=False):
        if if_test:
            image_features = self.image_encoder(image.type(self.dtype))

            prompts, prompts_double, temperature, spatial_T, rk_scale = self.prompt_learner()  # [n_cls, ctx_length, ctx_dim]
            tokenized_prompts = self.tokenized_prompts
            text_features = self.text_encoder(prompts, tokenized_prompts)

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # logit_scale = self.logit_scale.exp()
            logit_scale = 4.0
            logits = logit_scale * image_features @ text_features.t()

            return logits, None, None, None
        else:
            image_features = self.text_encoder(captions, None, if_embedding=False, if_sequence=False) 

            prompts, prompts_double, temperature, spatial_T, rk_scale = self.prompt_learner()  # [n_cls, ctx_length, ctx_dim]
            tokenized_prompts = self.tokenized_prompts
            text_features = self.text_encoder(prompts, tokenized_prompts)

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # logit_scale = self.logit_scale.exp()
            logit_scale = 4.0
            logits = logit_scale * image_features @ text_features.t()

            return logits, None, None, None

class DenseCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, return_interm_layers=False, nctx=None):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model, nctx)
        self.prompt_learner_m = PromptLearner(cfg, classnames, clip_model, nctx)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.text_encoder = TextEncoder(clip_model)

        self.model = clip_model
        self.return_interm_layers = return_interm_layers
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {"layer4": "0"}
        self.visual_encoder = IntermediateLayerGetter(self.model.visual, return_layers)
        self.positional_embedding = self.model.visual.attnpool.positional_embedding[1::]
        self.v_linear_weight = self.model.visual.attnpool.v_proj.weight
        self.v_linear_bias = self.model.visual.attnpool.v_proj.bias
        self.c_linear_weight = self.model.visual.attnpool.c_proj.weight
        self.c_linear_bias = self.model.visual.attnpool.c_proj.bias
        
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.cfg = cfg
        
        self.prompt_text_features = None
        
        self.model_pairs = [[self.prompt_learner,self.prompt_learner_m],
                           ]
        self.copy_params()
    
    def encode_image(self, x):
        def stem(x):
            for conv, bn in [(self.visual_encoder.conv1, self.visual_encoder.bn1), \
                (self.visual_encoder.conv2, self.visual_encoder.bn2), (self.visual_encoder.conv3, self.visual_encoder.bn3)]:
                x = self.visual_encoder.relu(bn(conv(x)))
            x = self.visual_encoder.avgpool(x)
            return x

        x = x.type(self.visual_encoder.conv1.weight.dtype)
        x = stem(x)
        x = self.visual_encoder.layer1(x)
        x = self.visual_encoder.layer2(x)
        x = self.visual_encoder.layer3(x)
        x = self.visual_encoder.layer4(x)
        return x
    
    def forward(self, image=None, captions=None, if_test=False, model_name='ema'):
        if if_test:        
            image_feat = self.encode_image(image)
            b, c, h, w = image_feat.shape
            x = image_feat.reshape(b, c, h * w).permute(2, 0, 1)
            # g_x = x.mean(0, keepdim=True)
            # x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW)xBxC        
            
            x = F.linear(x, self.v_linear_weight, self.v_linear_bias)
            x = F.linear(x, self.c_linear_weight, self.c_linear_bias)
            image_features = x

            image_feature_, _ = self.model.visual.attnpool(image_feat, if_pos=False)
            # ===============================================================

            # prompts, prompts_double, prompts_evidence, temperature, spatial_T, rk_scale = self.prompt_learner()
            # tokenized_prompts = self.tokenized_prompts
            # text_features = self.text_encoder(prompts, tokenized_prompts)
            # text_features_neg = self.text_encoder(prompts_double, tokenized_prompts)
            prompts, prompts_double, prompts_evidence, temperature, spatial_T, rk_scale = self.prompt_learner()
            if self.prompt_text_features is not None:
                prompt_text_features = self.prompt_text_features
                text_features = prompt_text_features['text_features']
                text_features_neg = prompt_text_features['text_features_neg']
                if 'text_features_evidence' in prompt_text_features:
                    text_features_evidence = prompt_text_features['text_features_evidence']
            else:
                tokenized_prompts = self.tokenized_prompts
                text_features = self.text_encoder(prompts, tokenized_prompts)
                text_features_neg = self.text_encoder(prompts_double, tokenized_prompts)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                text_features_neg = text_features_neg / text_features_neg.norm(dim=-1, keepdim=True)
                prompt_text_features = {'text_features': text_features, 'text_features_neg': text_features_neg}
                if self.cfg.TRAINER.Caption.use_evidence:
                    text_features_evidence = self.text_encoder(prompts_evidence, tokenized_prompts)
                    text_features_evidence = text_features_evidence / text_features_evidence.norm(dim=-1, keepdim=True)
                    
                    prompt_text_features['text_features_evidence'] = text_features_evidence
                self.prompt_text_features = prompt_text_features
            
            image_feature_ = image_feature_ / image_feature_.norm(dim=-1, keepdim=True)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            topk = 10
            sim_caption = (image_feature_ @ caption_text_feats.float().t())
            topk_sim_caption_scores, topk_sim_caption = sim_caption.topk(topk, -1)
            selected_caption_text_feats = caption_text_feats[topk_sim_caption.view(-1)].view(-1, topk, 1024).mean(1)
            image_feature_ = torch.cat([image_feature_[:, None], selected_caption_text_feats[:, None]], 1).mean(1)
            
            # text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            # text_features_neg = text_features_neg / text_features_neg.norm(dim=-1, keepdim=True)

            logit_scale = temperature.exp()  # rk_scale
            logit_scale = logit_scale if self.cfg.TRAIN.IF_LEARN_SCALE else 4.0 # 50
            logits_ = logit_scale * image_feature_ @ text_features.t()   # B * C,  cls * C, = B * cls
            logits_neg = image_features @ text_features_neg.t()    #  HW * B * C,  cls * C,  HW * B * cls
            
            tmp_scale = spatial_T.exp() if self.cfg.TRAIN.IF_LEARN_spatial_SCALE else self.cfg.TRAIN.spatial_SCALE_image  # 5 #
            if self.cfg.TRAINER.Caption.use_evidence:
                # text_features_evidence = self.text_encoder(prompts_evidence, tokenized_prompts)
                # text_features_evidence = text_features_evidence / text_features_evidence.norm(dim=-1, keepdim=True)
                logits_evidence = image_features @ text_features_evidence.t()
                # Winner-Take-All Regularization
                w = torch.nn.functional.softmax(tmp_scale * logits_neg * (logits_neg.max(-1)[0].unsqueeze(-1) + 1), -1)
                logits_neg = logits_neg * w
                prob_spatial = torch.nn.functional.softmax(logits_evidence * tmp_scale, dim=0)
            else:
                prob_spatial = torch.nn.functional.softmax(logits_neg * tmp_scale, dim=0)

            logits_local = torch.sum(logit_scale * logits_neg * prob_spatial, dim=0)

            return logits_, logits_local, logits_neg, image_features @ text_features.t(), topk_sim_caption_scores  # compare additional branch with global proxy
        else:
            image_feat = self.text_encoder(captions, None, if_embedding=False, if_sequence=True) 
            # b, l, d = image_feat.shape
            image_feature_ = image_feat[torch.arange(image_feat.shape[0]), captions.argmax(dim=-1)]  # BD
            image_features = image_feat.permute(1, 0, 2)  # LBD
            # ===============================================================

            prompts, prompts_double, prompts_evidence, temperature, spatial_T, rk_scale = self.prompt_learner()
            tokenized_prompts = self.tokenized_prompts
            text_features = self.text_encoder(prompts, tokenized_prompts)
            text_features_neg = self.text_encoder(prompts_double, tokenized_prompts)

            image_feature_ = image_feature_ / image_feature_.norm(dim=-1, keepdim=True)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_features_neg = text_features_neg / text_features_neg.norm(dim=-1, keepdim=True)
            
            # mask irrelavent tokens
            text_mask = (captions == 0).long() * (-10000)  # BL

            logit_scale = temperature.exp()  # rk_scale
            logit_scale = logit_scale if self.cfg.TRAIN.IF_LEARN_SCALE else 4.0 # 50 # temperature.exp()  # self.logit_scale.exp()
            logits_ = logit_scale * image_feature_ @ text_features.t()   # B * C,  cls * C, = B * cls
            logits_neg = image_features @ text_features_neg.t()    #  L * B * C,  cls * C,  L * B * cls
            logits_neg = logits_neg.permute(2, 1, 0) + text_mask[None, :, :]
            logits_neg = logits_neg.permute(2, 1, 0)
            
            tmp_scale = spatial_T.exp() if self.cfg.TRAIN.IF_LEARN_spatial_SCALE else self.cfg.TRAIN.spatial_SCALE_text  # 5 #
            if self.cfg.TRAINER.Caption.use_evidence:
                text_features_evidence = self.text_encoder(prompts_evidence, tokenized_prompts)
                text_features_evidence = text_features_evidence / text_features_evidence.norm(dim=-1, keepdim=True)
                logits_evidence = image_features @ text_features_evidence.t()    #  L * B * C,  cls * C,  L * B * cls
                logits_evidence = logits_evidence.permute(2, 1, 0) + text_mask[None, :, :]
                logits_evidence = logits_evidence.permute(2, 1, 0)
                # Winner-Take-All Regularization
                w = torch.nn.functional.softmax(tmp_scale * logits_neg * (logits_neg.max(-1)[0].unsqueeze(-1) + 1), -1)
                logits_neg = logits_neg * w
                prob_spatial = torch.nn.functional.softmax(logits_evidence * tmp_scale, dim=0)
            else:
                prob_spatial = torch.nn.functional.softmax(logits_neg * tmp_scale, dim=0)
           
            logits_local = torch.sum(logit_scale * logits_neg * prob_spatial, dim=0)
            
            if self.cfg.TRAIN.ema:
                with torch.no_grad():
                    self._momentum_update()
                    prompts_m, prompts_double_m, prompts_evidence_m, temperature, spatial_T, rk_scale = self.prompt_learner_m()
                    text_features_m = self.text_encoder(prompts_m, tokenized_prompts)
                    text_features_neg_m = self.text_encoder(prompts_double_m, tokenized_prompts)
                    text_features_m = text_features_m / text_features_m.norm(dim=-1, keepdim=True)
                    text_features_neg_m = text_features_neg_m / text_features_neg_m.norm(dim=-1, keepdim=True)
                    logits_m_ = logit_scale * image_feature_ @ text_features_m.t()   # B * C,  cls * C, = B * cls
                    logits_neg_m = image_features @ text_features_neg_m.t()    #  L * B * C,  cls * C,  L * B * cls
                    logits_neg_m = logits_neg_m.permute(2, 1, 0) + text_mask[None, :, :]
                    logits_neg_m = logits_neg_m.permute(2, 1, 0)
                    if self.cfg.TRAINER.Caption.use_evidence:
                        text_features_evidence_m = self.text_encoder(prompts_evidence_m, tokenized_prompts)
                        text_features_evidence_m = text_features_evidence_m / text_features_evidence_m.norm(dim=-1, keepdim=True)
                        logits_evidence_m = image_features @ text_features_evidence_m.t()    #  L * B * C,  cls * C,  L * B * cls
                        logits_evidence_m = logits_evidence_m.permute(2, 1, 0) + text_mask[None, :, :]
                        logits_evidence_m = logits_evidence_m.permute(2, 1, 0)
                        # Winner-Take-All Regularization
                        w = torch.nn.functional.softmax(tmp_scale * logits_neg_m * (logits_neg_m.max(-1)[0].unsqueeze(-1) + 1), -1)
                        logits_neg_m = logits_neg_m * w
                        prob_spatial_m = torch.nn.functional.softmax(logits_evidence_m * tmp_scale, dim=0)
                    else:
                        prob_spatial_m = torch.nn.functional.softmax(logits_neg_m * tmp_scale, dim=0)

                    logits_local_m = torch.sum(logit_scale * logits_neg_m * prob_spatial_m, dim=0)
            else:
                logits_m_, logits_local_m = None, None
            
            return logits_, logits_local, image_features, text_features, logits_m_, logits_local_m
        
    @torch.no_grad()    
    def copy_params(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient
    
    @torch.no_grad()        
    def _momentum_update(self):
        momentum = self.cfg.TRAIN.momentum
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * momentum + param.data * (1. - momentum)


# kl_loss = nn.KLDivLoss(reduction="batchmean")
# ce_loss = torch.nn.CrossEntropyLoss()

@TRAINER_REGISTRY.register()
class Caption_distill_double(TrainerX):
    def model_inference(self, input, name):
        return eval(f'self.model_{name}')(input, if_test=True, model_name=name)
        # return self.model(None, input)
    
    def before_epoch(self):
        if dist.get_rank() == 0:
            print(f'before_epoch: {self.epoch}')
        self.train_loader_x.sampler.set_epoch(self.epoch)

    def after_epoch(self):
        if dist.get_rank() == 0:
            print(f'after_epoch: {self.epoch}')
            last_epoch = (self.epoch + 1) == self.max_epoch
            do_test = not self.cfg.TEST.NO_TEST
            meet_checkpoint_freq = (
                (self.epoch + 1) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0
                if self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False
            )

            if meet_checkpoint_freq or last_epoch:
                self.save_model(self.epoch, self.output_dir)

    @torch.no_grad()
    def test(self, split=None, mode='train'):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
            print("Do evaluation on {} set".format(split))
        else:
            data_loader = self.test_loader
            print("Do evaluation on test set")

        names = self.get_model_names()
        if mode == 'test':
            all_output, all_output_pos, all_output_blocks, all_output_pos_blocks = {}, {}, {}, {}
            for name in names:
                all_output[name] = []
                all_output_pos[name] = []
                all_output_blocks[name] = []
                all_output_pos_blocks[name] = []
            
        def adjust_predictions(raw_predictions, normalized_cooccurrence_matrix, weight=1.0):  
            adjustment = torch.matmul(raw_predictions, normalized_cooccurrence_matrix)  
            adjusted_predictions = raw_predictions + weight * adjustment
        
            return adjusted_predictions
        
        import pickle
        result = pickle.load(open('freq_stats.pkl', 'rb'))
        
        # ================
        sims_all = []
        sims_blocks_all = []
        # ================
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label, input_blocks = self.parse_batch_test(batch)
            # output = self.model_inference(input)
            for name in names:
                output, output_pos, image_features_, text_features_, sim= self.model_inference(input, name)
                if self.cfg.TEST.use_freq:
                    # result['adj'][:, 0] = 0
                    p = torch.tensor(result['adj'] / result['nums'][:, np.newaxis], device='cuda', dtype=torch.float32)
                    p = p / p.sum(-1)[:,None]
                    output_pos = adjust_predictions(output_pos, p, 0.5)
                if mode=='test' and input_blocks is not None:
                    output_blocks = []
                    output_pos_blocks = []
                    sim_blocks = []
                    for input_block in input_blocks:
                        output_block, output_pos_block, image_features_, text_features_,sim_block = self.model_inference(input_block.reshape(-1, *input_block.shape[2:]), name)
                        output_block = output_block.reshape(input_block.shape[0], input_block.shape[1], -1)
                        #=================
                        sim_block = sim_block.reshape(input_block.shape[0], input_block.shape[1], -1)
                        sim_blocks.append(sim_block)
                        #==================
                        if output_pos_block is not None:
                            if self.cfg.TEST.use_freq:
                                output_pos_block = adjust_predictions(output_pos_block, p, 0.5)
                            output_pos_block = output_pos_block.reshape(input_block.shape[0], input_block.shape[1], -1)
                            output_pos_blocks.append(output_pos_block)
                        output_blocks.append(output_block)
                    #===================
                    sim_blocks = torch.cat(sim_blocks,dim=1)
                    # print("sim block shape:", sim_blocks.shape)
                    #====================
                    output_blocks = torch.cat(output_blocks, dim=1)
                    threshold = 0.3
                    alpha = output_blocks.max(dim=1)[0] # the maximum score of each class
                    beta =  output_blocks.min(dim=1)[0] # the minimum score of each class
                    gamma = (alpha > threshold).int()
                    s_ag = gamma * alpha + (1 - gamma) * beta
                    output_final = (1.4*s_ag + output)
                    if len(output_pos_blocks)>0:
                        output_pos_blocks = torch.cat(output_pos_blocks, dim=1)
                        output_pos_block = output_pos_block.reshape(input_block.shape[0], input_block.shape[1], -1)
                        threshold = 0.3
                        alpha = output_pos_blocks.max(dim=1)[0] # the maximum score of each class
                        beta =  output_pos_blocks.min(dim=1)[0] # the minimum score of each class
                        gamma = (alpha > threshold).int()
                        s_ag = gamma * alpha + (1 - gamma) * beta
                        output_pos_final = (1.4*s_ag + output_pos)
                else:
                    output_final = output
                    output_pos_final = output_pos
                        
                if mode == 'test':
                    # #=========================
                    # # sim append
                    # sims_all.append(sim.cpu()) # bs,5
                    # sims_blocks_all.append(sim_blocks.cpu()) # bs, 116, 5
                    # #==========================
                    all_output[name].append(output.cpu())
                    all_output_pos[name].append(output_pos.cpu())
                    if input_blocks is not None:
                        all_output_blocks[name].append(output_blocks.cpu())
                        all_output_pos_blocks[name].append(output_pos_blocks.cpu())
            
            if mode == 'test':
                #=========================
                # sim append (only save once)
                sims_all.append(sim.cpu()) # bs,5
                sims_blocks_all.append(sim_blocks.cpu()) # bs, 116, 5
                #==========================
           
            if mode == 'test' and len(names)>1:
                ###########
                ## fuse trategy need to add there when using multi model to evaluate
                ###########
                assert self.cfg.TEST.save_pth, "Can not use multi model when evaluating, fuse strtegy need to be added"
            self.evaluator.process(output_final, label, output_pos_final)
            
        if mode == 'test' and self.cfg.TEST.save_pth:
            try:
                #===============================
                sim_matrix = {}
                sim_matrix['sims_all'] = torch.cat(sims_all)
                sim_matrix['sims_blocks_all'] = torch.cat(sims_blocks_all)
                if not os.path.exists('./train_output/sim_matrix_B.pth'):
                    torch.save(sim_matrix, './train_output/sim_matrix_B.pth')
                #===============================
                need_save = {}
                for name in names:
                    need_save[name] = {}
                    need_save[name]['output'] = torch.cat(all_output[name])
                    need_save[name]['output_pos'] = torch.cat(all_output_pos[name])
                    if input_blocks is not None:
                        need_save[name]['output_blocks'] = torch.cat(all_output_blocks[name])
                        need_save[name]['output_pos_blocks'] = torch.cat(all_output_pos_blocks[name])
                
                torch.save(need_save, self.cfg.TEST.save_name)
            except:
                raise NotImplementedError
        
        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = "{}/{}".format(split, k)
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]

    def check_cfg(self, cfg):
        assert cfg.TRAINER.Caption.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        class_vars = self.__dict__
        cfg = self.cfg
        names = cfg.TEST.multi_model
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.Caption.PREC == "fp32" or cfg.TRAINER.Caption.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()
            
        for model_name in names:
            print(f'==================== Building model {model_name} in Caption_distill_double ======================')
            # nctx=64 if model_name == 'ema' else 16
            nctx=cfg.TRAINER.Caption.N_CTX
            print("Building custom CLIP")
            if self.cfg.TRAIN.MODEL == "DenseCLIP":
                class_vars[f'model_{model_name}'] = DenseCLIP(cfg, classnames, copy.deepcopy(clip_model), nctx=nctx)
            elif self.cfg.TRAIN.MODEL == "CustomCLIP":
                class_vars[f'model_{model_name}'] = CustomCLIP(cfg, classnames, copy.deepcopy(clip_model))
            else:
                raise NotImplementedError(f"model {self.cfg.TRAIN.MODEL} not implemented")

            print("Turning off gradients in both the image and the text encoder")
            for name, param in eval(f'self.model_{model_name}').named_parameters():
                if "prompt_learner" not in name:
                    param.requires_grad_(False)

            # load_pretrained_weights(self.model.prompt_learner, 'output/voc2007_caption_distill_abinf/Caption_distill_double/rn50_fixscale/nctx16_cscFalse_ctpend/seed3/prompt_learner/model-best.pth.tar')
            if cfg.MODEL.INIT_WEIGHTS:
                load_pretrained_weights(eval(f'self.model_{model_name}').prompt_learner, cfg.MODEL.INIT_WEIGHTS)

            eval(f'self.model_{model_name}').to(self.device)
            # NOTE: only give prompt_learner to the optimizer
            self.optim = build_optimizer(eval(f'self.model_{model_name}').prompt_learner, cfg.OPTIM)
            self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
            self.register_model(model_name, eval(f'self.model_{model_name}').prompt_learner, self.optim, self.sched)
            # self.register_model("prompt_learner2", self.model.prompt_learner, self.optim, self.sched)

            self.scaler = GradScaler() if cfg.TRAINER.Caption.PREC == "amp" else None

            # Note that multi-gpu training could be slow because CLIP's size is
            # big, which slows down the copy operation in DataParallel
            # device_count = torch.cuda.device_count()
            # if device_count > 1:
            #     print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            #     self.model = nn.DataParallel(self.model)
            device = torch.device("cuda", dist.get_rank())
            class_vars[f'model_{model_name}'] = DDP(eval(f'self.model_{model_name}'), device_ids=[device], output_device=device, find_unused_parameters=True)  
        
    def forward_backward(self, batch):
        name = self.get_model_names()[0]
        image, label = self.parse_batch_train(batch)
        kl_loss = nn.KLDivLoss(reduction="batchmean")
        
        prec = self.cfg.TRAINER.Caption.PREC
        if prec == "amp":
            with autocast():
                output, output_local, _, _ = eval(f'self.model_{name}')(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output, output_local, _, _, output_m, output_local_m = eval(f'self.model_{name}')(None, image)
            if self.cfg.TRAIN.LOSSFUNC == 'double_ranking':
                r_loss = ranking_loss(output, label, scale_ = 1.0, margin_ = 1)
                if output_local is not None:
                    r_loss += ranking_loss(output_local, label, scale_ = 1.0, margin_ = 1)
                if output_m is not None:
                    ema_loss = kl_loss(F.log_softmax(output, dim=-1), F.softmax(output_m, dim=-1)) + \
                                kl_loss(F.log_softmax(output_local, dim=-1), F.softmax(output_local_m, dim=-1)) * 10000
                    # ema_loss = -torch.sum(F.log_softmax(output, dim=1)*F.softmax(output_m,dim=1), dim=1).mean()
                    loss = r_loss + ema_loss
                else:
                    loss = r_loss
            elif self.cfg.TRAIN.LOSSFUNC == 'soft_ce':
                loss = soft_cross_entropy(output, label)
            elif self.cfg.TRAIN.LOSSFUNC == 'dbl': # Distribution-Balanced Loss
                caption_feat_root = os.getcwd()
                caption_feat_root = osp.join(caption_feat_root, 'generated_captions/')
                freq_file = osp.join(caption_feat_root, f'{self.cfg.TRAIN.Caption_name}_class_freq.pkl')
                loss_function = ResampleLoss(
                    use_sigmoid=True,
                    reweight_func='rebalance',
                    focal=dict(focal=False, balance_param=2.0, gamma=2),
                    logit_reg=dict(),
                    map_param=dict(alpha=0.1, beta=10.0, gamma=0.2),
                    loss_weight=1.0, freq_file=freq_file
                )
                # loss_function = ResampleLoss(
                #     use_sigmoid=True,
                #     reweight_func='rebalance',
                #     focal=dict(focal=True, balance_param=2.0, gamma=2),
                #     logit_reg=dict(neg_scale=2.0, init_bias=0.05),
                #     map_param=dict(alpha=0.1, beta=10.0, gamma=0.2),
                #     loss_weight=1.0, freq_file=freq_file
                # )
                if output_local is None:
                    loss = loss_function(output, label)
                else:
                    loss = loss_function(output, label) + loss_function(output_local, label)
            elif self.cfg.TRAIN.LOSSFUNC == 'ranking_with_cooccurrence':
                import pickle
                result = pickle.load(open('freq_stats.pkl', 'rb'))
                p = torch.tensor(result['adj'] / result['nums'][:, np.newaxis], device='cuda', dtype=torch.float32)
                p = p / p.sum(-1)[:,None]
                r_loss = ranking_loss_with_cooccurrence(output, label, p, scale_ = 1.0, margin_ = 1)
                if output_local is not None:
                    r_loss += ranking_loss_with_cooccurrence(output_local, label, p, scale_ = 1.0, margin_ = 1)
                loss = r_loss
            else:
                raise NotImplementedError(f'loss function {self.cfg.TRAIN.LOSSFUNC} not implemented')

            if output_m is None:
                loss_summary = {
                        f'loss_{self.cfg.TRAIN.LOSSFUNC}': loss.item(),
                }
            else:
                loss_summary = {
                        'r_loss': r_loss.item(),
                        'ema_loss': ema_loss.item(),
                }
            if self.cfg.TRAIN.TRAINING_METHOD.NAME == "lmpt":
                caption_feat_root = os.getcwd()
                caption_feat_root = osp.join(caption_feat_root, 'generated_captions/')
                freq_file = osp.join(caption_feat_root, f'{self.cfg.TRAIN.Caption_name}_class_freq.pkl')
                class_weights = torch.from_numpy(np.asarray(mmcv.load(freq_file)['class_freq'])).to(torch.float32).cuda()
                hinge_loss = SoftMarginHingeEmbeddingLoss(margin=0.2, class_counts=class_weights)
                
                if isinstance(self.model, DDP):
                    captions_ = self.model.module.model.token_embedding(image)
                    text_features = self.model.module.prompt_learner()[0]
                else:
                    captions_ = self.model.model.token_embedding(image)
                    text_features = self.model.prompt_learner()[0]
                m_ctx = self.cfg.TRAINER.Caption.M_CTX
                a = captions_[:,:(77-m_ctx),:].unsqueeze(1).expand(captions_.shape[0], label.shape[1], 77-m_ctx, captions_.shape[-1]).to(torch.float32).cuda()
                b = text_features[:,m_ctx:,:].unsqueeze(0).expand(captions_.shape[0], label.shape[1], 77-m_ctx, captions_.shape[-1]).to(torch.float32).cuda()
                x = 1 - torch.cosine_similarity(a, b, dim=-1) 
                y = 2 * label.unsqueeze(2).expand(label.shape[0], label.shape[1], 77-m_ctx) - 1

                loss_2 = hinge_loss(x, y)
                loss_summary.update({
                    f'loss_lmpt': loss_2.item()
                })
                loss = self.cfg.TRAIN.TRAINING_METHOD.LAMBDA * loss + (1 - self.cfg.TRAIN.TRAINING_METHOD.LAMBDA) * loss_2

            self.model_backward_and_update(loss)

        loss_summary.update({
            "loss": loss.item(),
        })

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
            # print(state_dict)


def customContraLoss(y_pred, y_true, tau, eps=1e-6):
    logits = y_pred / tau + eps
    logits = logits - torch.max(logits, dim=1, keepdim=True)[0]
    exp_logits = torch.exp(logits)
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + eps)
    mean_log_prob_pos = (y_true * log_prob).sum(1) / (y_true.sum(1) + eps)
    return -mean_log_prob_pos.mean()