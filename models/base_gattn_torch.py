import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_softmax_cross_entropy(logits, labels, mask):
    loss = F.cross_entropy(logits, torch.argmax(labels, dim=1), reduction='none')
    mask = mask.float()
    mask = mask / torch.mean(mask)
    loss = loss * mask
    return torch.mean(loss)


def masked_accuracy(logits, labels, mask):
    correct_prediction = (torch.argmax(logits, dim=1) == torch.argmax(labels, dim=1)).float()
    mask = mask.float()
    mask = mask / torch.mean(mask)
    accuracy_all = correct_prediction * mask
    return torch.mean(accuracy_all)
