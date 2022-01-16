import torch
import torch.nn as nn


# This loss converges much slower than cross_entropy
# The begining results (MRR) are very low, it will get much better after 100-200 epochs 
def bce_loss(pred, soft_targets):
    loss_func = nn.BCELoss() 
    return loss_func(torch.sigmoid(pred)+1e-8, soft_targets)


def cross_entropy(pred, soft_targets):
    logsoftmax = nn.LogSoftmax(dim=1)
    return torch.mean(torch.sum(- soft_targets * logsoftmax(pred + 1e-8), 1))
