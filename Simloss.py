import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def dotsimilarity(x1, x2):
    x1_flat = torch.flatten(x1)
    x2_flat = torch.flatten(x2)

    return torch.dot(x1_flat, x2_flat)



def SimLoss (self, pred_u1, pred_u2, batch_size, temparature, k):
    # unlabeled_train_iter = iter(unlabel_loader)
    pred_u1 = F.normalize(pred_u1, dim=1)
    pred_u2 = F.normalize(pred_u2, dim=1)
    pred_concat = torch.cat((pred_u1, pred_u2), dim=0)

    sim = nn.CosineSimilarity(dim=-1)
    sim_mat = sim(pred_concat.unsqueeze(1), pred_concat.unsqueeze(0)) #(2*batch_size) * (2*batch_size)
    pos1_mask = torch.diag(sim_mat, batch_size)
    pos2_mask = torch.diag(sim_mat, -batch_size)


def SimLoss2(self, pred_list, batch_size, T):
    k = len(pred_list)
    # concatenate predicted results for augmented unlabeled data
    pred_concat = torch.cat(pred_list, dim=0)

    sim = nn.CosineSimilarity(dim=-1)
    sim_mat = sim(pred_concat.unsqueeze(1), pred_concat.unsqueeze(0)) #(k*batch_size) * (k*batch_size)
    
    # get mask matrix that can retreive positive pairs in the similiarity matrix.
    pos_mask_np = np.zeros(((k * batch_size), (k * batch_size)))
    for i in range(1, k):
        pos1 = np.eye((k * batch_size), k= i*batch_size)
        pos2 = np.eye((k * batch_size), k= -i*batch_size)
        pos_mask_np += pos1 + pos2
    pos_mask = torch.from_numpy(pos_mask_np)
    pos_mask = pos_mask.type(torch.bool)
    pos_mask.to(torch.device('cuda'))
    
    # get mask matrix that can retreive negative pairs in the similiarity matrix.
    neg_mask_np = np.ones(((k * batch_size), (k * batch_size)))
    neg_mask_np -= pos_mask_np
    neg_mask_np -= np.eye(k * batch_size)
    neg_mask = torch.from_numpy(neg_mask_np)
    neg_mask = neg_mask.type(torch.bool)
    neg_mask.to(torch.device('cuda'))
    
    # multiply mask to similiarity matrix to get pairs
    pos_pairs = sim_mat[pos_mask].view(2 * batch_size, -1)
    neg_pairs = sim_mat[neg_mask].view(2 * batch_size, -1)

    logits = torch.cat((pos_pairs, neg_pairs), dim=1) / T
    labels = torch.zeros(k * batch_size)
    labels.to(torch.device('cuda'))

    criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    # SIMCLR gets average loss
    loss = criterion(logits, labels) / k*batch_size

    return loss

class NT_Xent(nn.Module):
    def __init__(self, batch_size, temperature):
        super(NT_Xent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.similarity = torch.nn.CosineSimilarity(dim=-1)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(torch.device('cuda'))

    def forward(self, pred_u1, pred_u2):
        # add loss function here
        pred_u1 = F.normalize(pred_u1, dim=1)
        pred_u2 = F.normalize(pred_u2, dim=1)
        pred_concat = torch.cat((pred_u1, pred_u2), dim=0)
        similarity_matrix = self.similarity(pred_concat.unsqueeze(1), pred_concat.unsqueeze(0))

        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(torch.device('cuda')).long()
        loss = self.criterion(logits, labels)

        return loss / (2*self.batch_size)