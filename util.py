import math
import os
import random

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset

cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


def sample_triplets(embeddings, labels, margin=2.0):
    """
    Batch 内 Hard Negative Mining
    """
    device = embeddings.device
    dist_mat = torch.cdist(embeddings, embeddings, p=2)

    labels_np = labels.cpu().numpy()
    unique_labels = np.unique(labels_np)

    anc_list, pos_list, neg_list = [], [], []

    for i in range(len(embeddings)):
        label = labels_np[i]

        pos_mask = (labels == label)
        pos_mask[i] = False
        if not pos_mask.any(): continue
        pos_idx = torch.argmax(dist_mat[i][pos_mask])
        actual_pos_idx = torch.where(pos_mask)[0][pos_idx]

        neg_mask = (labels != label)
        if not neg_mask.any(): continue
        neg_idx = torch.argmin(dist_mat[i][neg_mask])
        actual_neg_idx = torch.where(neg_mask)[0][neg_idx]

        if dist_mat[i, actual_pos_idx] + margin > dist_mat[i, actual_neg_idx]:
            anc_list.append(embeddings[i])
            pos_list.append(embeddings[actual_pos_idx])
            neg_list.append(embeddings[actual_neg_idx])

    if not anc_list: return None, None, None
    return torch.stack(anc_list), torch.stack(pos_list), torch.stack(neg_list)


def save_checkpoint(state, save):
    if not os.path.exists(save):
        os.makedirs(save)
    filename = os.path.join(save, 'model_best.pth.tar')
    torch.save(state, filename)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_label_names(index, classify_dim):
    output_v = []
    label_num = pd.Categorical(index).codes
    for i in range(classify_dim):
        temp = index[np.array(np.where(label_num == i)[0][0])]
        output_v.append(temp)
    return output_v


def to_tensor(X_train, X_test, Y_train, Y_test):
    X_train = torch.from_numpy(X_train)
    X_train = Variable(X_train.type(FloatTensor))
    X_test = torch.from_numpy(X_test)
    X_test = Variable(X_test.type(FloatTensor))
    Y_train = torch.from_numpy(Y_train)
    Y_train = Y_train.type(LongTensor)
    Y_test = torch.from_numpy(Y_test)
    Y_test = Y_test.type(LongTensor)
    return X_train, X_test, Y_train, Y_test


class SingleCellDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        img, target = self.data[index, :], self.label[index]
        sample = {'data': img, 'label': target}
        return sample

    def __len__(self):
        return len(self.data)


def normalize_adata(adata):
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.scale(adata)
    return adata


def get_encodings(model, dl):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    shared_list = []
    label_list = []
    ori_data_list = []

    with torch.no_grad():
        for batch_sample in dl:
            x = batch_sample['data'].to(device).float()
            x_flat = x.view(x.size(0), -1)
            labels = batch_sample['label']

            outputs = model(x_flat)

            shared_list.append(outputs[7].detach().cpu())
            ori_data_list.append(x_flat.detach().cpu())
            label_list.append(labels.detach().cpu())

    return {
        'shared': torch.cat(shared_list, dim=0),
        'ori': torch.cat(ori_data_list, dim=0),
        'label': torch.cat(label_list, dim=0)
    }


def get_decodings(model, dl):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    decodings = []
    ori_data = []

    with torch.no_grad():
        for i, batch_sample in enumerate(dl):
            x = batch_sample['data'].to(device).float()
            x = torch.reshape(x, (x.size(0), -1))

            outputs = model(x)
            x_recon = outputs[0]

            decodings.append(x_recon.cpu())
            ori_data.append(x.cpu())

    return torch.cat(decodings, dim=0), torch.cat(ori_data, dim=0)

class BaseLoss:
    def __init__(self, weight=1.0):
        self.weight = weight

class ReconstructionLoss(BaseLoss):
    def __init__(self, config):
        loss_weight = config.get('train_loss_weight', 1.0)
        super().__init__(weight=loss_weight)

    def __call__(self, reconstruction, raw_data):
        loss = F.mse_loss(reconstruction, raw_data)
        return loss * self.weight

class WeightedCrossEntropyLabelSmooth(nn.Module):
    def __init__(self, epsilon=0.1, weights=None):
        super(WeightedCrossEntropyLabelSmooth, self).__init__()
        self.epsilon = epsilon
        self.register_buffer('weights', weights)

    def forward(self, inputs, targets):
        return F.cross_entropy(
            inputs,
            targets,
            weight=self.weights,
            label_smoothing=self.epsilon
        )

class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes=17, epsilon=0.1):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss

class scMoDCA_CMDL_LossManager:
    def __init__(self, config):
        self.recon_criterion = ReconstructionLoss(config)
        self.cls_criterion = CrossEntropyLabelSmooth(num_classes=config.get('num_classes', 22))

    def main_loss(self, model, x, results, y_true, epoch, dataset):
        x_res, z, spec1, spec2, shared, orig_m1, orig_m2, mu_unnorm, x_cluster = results
        x_recon = model.decoder(z, spec1, spec2)
        l_recon = self.recon_criterion(x_recon, x)
        mi_cond_1to2 = model.club_cond_1(torch.cat([spec1, shared], dim=-1), spec2)
        mi_cond_2to1 = model.club_cond_2(torch.cat([spec2, shared], dim=-1), spec1)
        mi_cond_symmetric = mi_cond_1to2 + mi_cond_2to1
        mi_basic = model.club1(spec1, shared) + model.club2(spec2, shared)
        l_disentangle = torch.relu(mi_basic + mi_cond_symmetric * 2.0)
        mi_preserve = model.infonce_mod1_shared(orig_m1, shared) + model.infonce_mod1_shared(orig_m2, shared)
        l_preserve = -mi_preserve
        l_mi_final = (l_disentangle + torch.clamp(l_preserve, min=-2.0))
        l_cls = self.cls_criterion(x_cluster, y_true)
        w = {
            'recon': 1.0,
            'mi': 0,
            'cls': 1
        }
        base_mi_weight = 0.05
        if epoch <= 3:
            w['mi'] = 0.0
        elif epoch <= 15:
            w['mi'] = base_mi_weight * ((epoch - 3) / 12.0)
        else:
            w['mi'] = base_mi_weight

        if epoch > 15:
            w['cls'] = 2.0

        total_loss = (l_recon * w['recon']) + \
                     (l_mi_final * w['mi']) + \
                     l_cls * w['cls']
        return total_loss, (l_recon, l_mi_final, l_cls)

    def learn_losses(self, model, results):
        spec1, spec2, shared = results[2].detach(), results[3].detach(), results[4].detach()

        l_club_learn = model.club1.learning_loss(spec1, shared) + \
                       model.club2.learning_loss(spec2, shared)

        l_cond_club_learn = model.club_cond_1.learning_loss(torch.cat([spec1, shared], dim=-1), spec2) + \
                            model.club_cond_2.learning_loss(torch.cat([spec2, shared], dim=-1), spec1)

        total_critic_loss = l_club_learn + l_cond_club_learn

        with torch.no_grad():
            mi_val_1to2 = model.club_cond_1(torch.cat([spec1, shared], dim=-1), spec2).item()
            mi_val_2to1 = model.club_cond_2(torch.cat([spec2, shared], dim=-1), spec1).item()
            current_mi = (mi_val_1to2 + mi_val_2to1) / 2.0

        return total_critic_loss, current_mi
