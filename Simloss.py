import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def dotsimilarity(x1, x2):
    x1_flat = torch.flatten(x1)
    x2_flat = torch.flatten(x2)

    return torch.dot(x1_flat, x2_flat)  
      
class SimLoss(object):
    def __call__(self, pred_list, batch_size, T):
        k = len(pred_list)
        # concatenate predicted results for augmented unlabeled data
        for i in range(k):
            pred_list[i] = F.normalize(pred_list[i], dim=1)
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
        pos_pairs = sim_mat[pos_mask].view(k * batch_size, -1)
        neg_pairs = sim_mat[neg_mask].view(k * batch_size, -1)

        logits = torch.cat((pos_pairs, neg_pairs), dim=1) / T
        labels = torch.zeros(k * batch_size).to(torch.device('cuda')).long()
        logits.to(torch.device('cuda'))

        criterion = torch.nn.CrossEntropyLoss(reduction="sum")

        # SIMCLR gets average loss
        loss = criterion(logits, labels) / (k*batch_size)

        return loss