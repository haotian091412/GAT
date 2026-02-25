import torch
import torch.nn as nn
import torch.nn.functional as F


class AttnHead(nn.Module):
    def __init__(self, in_sz, out_sz, activation=F.elu, in_drop=0.0, coef_drop=0.0, residual=False):
        super(AttnHead, self).__init__()
        self.in_drop = in_drop
        self.coef_drop = coef_drop
        self.residual = residual
        self.activation = activation

        self.conv_seq = nn.Conv1d(in_sz, out_sz, 1, bias=False)
        self.conv_f1 = nn.Conv1d(out_sz, 1, 1)
        self.conv_f2 = nn.Conv1d(out_sz, 1, 1)

        self.bias = nn.Parameter(torch.zeros(1, 1, out_sz))

        self.in_dropout = nn.Dropout(in_drop)
        self.coef_dropout = nn.Dropout(coef_drop)

        if residual:
            if in_sz != out_sz:
                self.residual_conv = nn.Conv1d(in_sz, out_sz, 1, bias=False)
            else:
                self.residual_conv = None

        nn.init.xavier_uniform_(self.conv_seq.weight)
        nn.init.xavier_uniform_(self.conv_f1.weight)
        nn.init.xavier_uniform_(self.conv_f2.weight)

    def forward(self, seq, bias_mat):
        seq = self.in_dropout(seq)

        seq_fts = self.conv_seq(seq.transpose(1, 2)).transpose(1, 2)

        f_1 = self.conv_f1(seq_fts.transpose(1, 2)).transpose(1, 2)
        f_2 = self.conv_f2(seq_fts.transpose(1, 2)).transpose(1, 2)

        logits = f_1 + f_2.transpose(1, 2)
        coefs = F.leaky_relu(logits) + bias_mat
        coefs = F.softmax(coefs, dim=-1)
        coefs = self.coef_dropout(coefs)

        seq_fts = self.in_dropout(seq_fts)

        vals = torch.matmul(coefs, seq_fts)
        ret = vals + self.bias

        if self.residual:
            if self.residual_conv is not None:
                ret = ret + self.residual_conv(seq.transpose(1, 2)).transpose(1, 2)
            else:
                ret = ret + seq

        return self.activation(ret)


class AvgHead(nn.Module):
    def __init__(self, in_sz, out_sz, activation=F.elu, in_drop=0.0, residual=False):
        super(AvgHead, self).__init__()
        self.in_drop = in_drop
        self.residual = residual
        self.activation = activation

        self.conv_seq = nn.Conv1d(in_sz, out_sz, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(1, 1, out_sz))

        self.in_dropout = nn.Dropout(in_drop)

        if residual:
            if in_sz != out_sz:
                self.residual_conv = nn.Conv1d(in_sz, out_sz, 1, bias=False)
            else:
                self.residual_conv = None

        nn.init.xavier_uniform_(self.conv_seq.weight)

    def forward(self, seq, adj_mask):
        seq = self.in_dropout(seq)
        seq_fts = self.conv_seq(seq.transpose(1, 2)).transpose(1, 2)

        coefs = adj_mask / (adj_mask.sum(dim=-1, keepdim=True) + 1e-10)

        seq_fts = self.in_dropout(seq_fts)

        vals = torch.matmul(coefs, seq_fts)
        ret = vals + self.bias

        if self.residual:
            if self.residual_conv is not None:
                ret = ret + self.residual_conv(seq.transpose(1, 2)).transpose(1, 2)
            else:
                ret = ret + seq

        return self.activation(ret)
