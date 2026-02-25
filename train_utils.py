import torch
import torch.nn.functional as F


def masked_softmax_cross_entropy(logits, labels, mask):
    loss = F.cross_entropy(logits, torch.argmax(labels, dim=1), reduction='none')
    mask = mask.float()
    mask /= mask.mean()
    loss *= mask
    return loss.mean()


def masked_accuracy(logits, labels, mask):
    pred = torch.argmax(logits, dim=1)
    correct = (pred == torch.argmax(labels, dim=1)).float()
    mask = mask.float()
    mask /= mask.mean()
    correct *= mask
    return correct.mean()


def train_step(model, optimizer, features, bias_mat, labels, mask, l2_coef, attn_drop, ffd_drop, device, use_avg=False):
    model.train()
    optimizer.zero_grad()

    logits = model(features, bias_mat)

    log_resh = logits.reshape(-1, logits.size(-1))
    lab_resh = labels.reshape(-1, labels.size(-1))
    msk_resh = mask.reshape(-1)

    loss = masked_softmax_cross_entropy(log_resh, lab_resh, msk_resh)

    l2_loss = 0.0
    for name, param in model.named_parameters():
        if 'bias' not in name:
            l2_loss += torch.norm(param, 2)
    loss = loss + l2_coef * l2_loss

    acc = masked_accuracy(log_resh, lab_resh, msk_resh)

    loss.backward()
    optimizer.step()

    return loss.item(), acc.item()


def eval_step(model, features, bias_mat, labels, mask, device, use_avg=False):
    model.eval()

    with torch.no_grad():
        logits = model(features, bias_mat)

        log_resh = logits.reshape(-1, logits.size(-1))
        lab_resh = labels.reshape(-1, labels.size(-1))
        msk_resh = mask.reshape(-1)

        loss = masked_softmax_cross_entropy(log_resh, lab_resh, msk_resh)
        acc = masked_accuracy(log_resh, lab_resh, msk_resh)

    return loss.item(), acc.item()
