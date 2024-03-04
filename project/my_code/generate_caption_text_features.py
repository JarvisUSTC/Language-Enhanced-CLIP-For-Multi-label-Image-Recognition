import pickle5 as pickle
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

with open("../output/text_result/generated_captions/ChatGLM_multi_labels_filtered_all_caption_tokenized.pkl", 'rb') as f:
    prompts = pickle.load(f)
    
with open("./generated_captions/ChatGLM_multi_labels_filtered_labels.pkl", "rb") as f:
    word_based_caption = pickle.load(f)
    
    
sample_capid = word_based_caption.keys()
sample_capid_inverse_idx = {}
for i, j in enumerate(sample_capid):
    sample_capid_inverse_idx[j] = i
    
train = []
for capid in word_based_caption:
    i = sample_capid_inverse_idx[capid]
    # print("prompt shape:",prompts[i].shape)
    item_ = (prompts[i], torch.tensor(word_based_caption[capid]))
    train.append(item_)
    
    
import os
from clip import clip
import torch.nn as nn

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

def load_clip_to_cpu(backbone_name="RN50"):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url, root= os.path.expanduser(f"~/.cache/clip_0"))

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model

clip_model = load_clip_to_cpu()
text_encoder = TextEncoder(clip_model).cuda().eval()

batch_size = 256  # you can adjust this value according to your GPU memory  
train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
all_text_feats = []
all_labels = []
with torch.no_grad():
    for (caption, label) in tqdm(train_loader):
        text_feat = text_encoder(caption.to("cuda"), None, if_embedding=False, if_sequence=True).cpu() 
        text_feat = text_feat[torch.arange(text_feat.shape[0]), caption.argmax(dim=-1)]  # BD
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        all_text_feats.append(text_feat)
        all_labels.append(label)
        torch.cuda.empty_cache()

all_text_feats = torch.cat(all_text_feats, dim=0)
all_labels = torch.cat(all_labels, dim=0)
# save to pickle
with open("./ChatGLM_multi_labels_filtered_22w_all_caption_text_feats.pkl", "wb") as f:
    pickle.dump(all_text_feats, f)