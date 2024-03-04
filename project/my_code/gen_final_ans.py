import torch
import numpy as np
import json
import copy

def sigma(data):
    """
    data: [13965, 116, 80]
    return var:[13965, 116]
    """
    var_matrix = torch.var(data,dim=2)
    return var_matrix

sim_matrix = torch.load("./train_output/sim_matrix_B.pth")
sims_scores = sim_matrix['sims_blocks_all']
print(sims_scores.shape)

def fuse(data, threshold=0.2):
    # max mode
    sims_mat = sims_scores.mean(-1,keepdim=True)
    # mean mode
    # sims_mat = sims_scores[:,:,0].unsqueeze(-1)

    sims_mat = torch.ones_like(sims_mat) + sims_mat 
    data = (sims_mat * data)

    var_matrix = torch.var(data,dim=2)
    var_matrix = var_matrix.unsqueeze(-1)

    var_matrix = torch.ones_like(var_matrix) + var_matrix
    data = (var_matrix * data)

    alpha = data.max(dim=1)[0] # the maximum score of each class
    beta =  data.min(dim=1)[0] # the minimum score of each class
    gamma = (alpha > threshold).int()
    s_ag = gamma * alpha + (1 - gamma) * beta
    return s_ag

def fuse6(data, threshold=0.2):
    # max mode
    sims_mat = sims_scores.mean(-1,keepdim=True)
    # mean mode
    # sims_mat = sims_scores[:,:,0].unsqueeze(-1)

    # for origin data

    var_matrix0 = torch.var(data,dim=2)
    var_matrix0 = var_matrix0.unsqueeze(-1)

    var_matrix0 = torch.ones_like(var_matrix0) + var_matrix0
    data_var = (var_matrix0 * data)

    # for origin data
    sims_mat = torch.ones_like(sims_mat) + sims_mat 
    data_sim = (sims_mat * data)

    # data = var_matrix * var_matrix * data_var
    # data = var_matrix * var_matrix * data_sim

    data_ens = data_sim
    var_matrix1 = torch.var(data_ens,dim=2)
    var_matrix1 = var_matrix1.unsqueeze(-1)
    var_matrix1 = torch.ones_like(var_matrix1) + var_matrix1
    
    data = var_matrix0 * var_matrix1 * data_ens

    alpha = data.max(dim=1)[0] # the maximum score of each class
    beta =  data.min(dim=1)[0] # the minimum score of each class
    gamma = (alpha > threshold).int()
    s_ag = gamma * alpha + (1 - gamma) * beta
    return s_ag

# load inferenced data
data = torch.load("./train_output/data.pth", map_location='cpu')
data_ema = torch.load("./train_output/data_ema.pth", map_location='cpu')
data['ema'] = data_ema['ema']

data_best = torch.load("./train_output/data_evidence.pth", map_location='cpu')
data['best'] = data_best['best']
data['difft'] = data_best['difft']

# read data from 8 inferenced models
best_output = data['best']['output'].cpu()
best_aux = data['best']['output_pos'].cpu()
best_outputb = data['best']['output_blocks'].cpu()
best_auxb = data['best']['output_pos_blocks'].cpu()

ema_output = data['ema']['output'].cpu()
ema_aux = data['ema']['output_pos'].cpu()
ema_outputb = data['ema']['output_blocks'].cpu()
ema_auxb = data['ema']['output_pos_blocks'].cpu()

zema_output = data['zema']['output'].cpu()
zema_aux = data['zema']['output_pos'].cpu()
zema_outputb = data['zema']['output_blocks'].cpu()
zema_auxb = data['zema']['output_pos_blocks'].cpu()

diff_output = data['diff']['output'].cpu()
diff_aux = data['diff']['output_pos'].cpu()
diff_outputb = data['diff']['output_blocks'].cpu()
diff_auxb = data['diff']['output_pos_blocks'].cpu()

difft_output = data['difft']['output'].cpu()
difft_aux = data['difft']['output_pos'].cpu()
difft_outputb = data['difft']['output_blocks'].cpu()
difft_auxb = data['difft']['output_pos_blocks'].cpu()

diffh_output = data['diffh']['output'].cpu()
diffh_aux = data['diffh']['output_pos'].cpu()
diffh_outputb = data['diffh']['output_blocks'].cpu()
diffh_auxb = data['diffh']['output_pos_blocks'].cpu()


# calculate logits of each model
# use results of block_size=2,3,4
coef = 1.5
aux_coef = 1.5
best_o = best_output + coef * fuse6(best_outputb[:, :])
best_a = best_aux + coef * fuse6(best_auxb[:, :])
best_res = best_o + aux_coef * best_a

ema_o = ema_output + coef*fuse(ema_outputb[:, :])
ema_a = ema_aux + coef*fuse(ema_auxb[:, :])
ema_res = ema_o + ema_a

zema_o = zema_output + coef*fuse(zema_outputb[:, :])
zema_a = zema_aux + coef*fuse(zema_auxb[:, :])
zema_res = zema_o + zema_a

diff_o = diff_output + coef * fuse(diff_outputb[:, :])
diff_a = diff_aux + coef * fuse(diff_auxb[:, :])
diff_res = diff_o + diff_a

diffh_o = diffh_output + coef * fuse(diffh_outputb[:, :])
diffh_a = diffh_aux + coef * fuse(diffh_auxb[:, :])
diffh_res = diffh_o + diffh_a

difft_o = difft_output + coef * fuse(difft_outputb[:, :])
difft_a = difft_aux + coef * fuse(difft_auxb[:, :])
difft_res = difft_o + difft_a

# good classes
ema_classes = [2, 6, 7, 8, 14, 16, 17, 25, 27, 31, 33, 34, 37, 38, 39, 40, 41, 43, 49, 52, 57, 62, 67, 73, 74, 76]
zema_classes = [0, 4, 21, 23, 32, 35, 45, 53, 54, 55, 58, 59, 61]
diff_classes = [13, 22, 42, 78]
diffh_classes = [24, 26, 47, 56]
difft_classes = [1, 3, 12, 29, 36, 68, 72, 79]

# # fuse results
fuse_res = copy.deepcopy(best_res)
fuse_res[:, ema_classes] = ema_res[:, ema_classes]
fuse_res[:, zema_classes] = zema_res[:, zema_classes]
fuse_res[:, diff_classes] = diff_res[:, diff_classes]
fuse_res[:, diffh_classes] = diffh_res[:, diffh_classes]
fuse_res[:, difft_classes] = difft_res[:, difft_classes]

# save results
ret_list = []
for i in fuse_res:
    ret_list.append(i.tolist())
with open('../output/impreds.json', 'w') as f:
    json.dump(ret_list, f)