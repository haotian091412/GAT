import torch
import torch.nn as nn
import torch.nn.functional as F


class AttnHead(nn.Module):
    def __init__(self, in_sz, out_sz, activation=F.elu, in_drop=0.0, coef_drop=0.0, residual=False):
        super(AttnHead, self).__init__()
        self.in_sz = in_sz
        self.out_sz = out_sz
        self.activation = activation
        self.in_drop = in_drop
        self.coef_drop = coef_drop
        self.residual = residual

        self.conv_seq = nn.Conv1d(in_sz, out_sz, 1, bias=False)
        self.conv_f1 = nn.Conv1d(out_sz, 1, 1)
        self.conv_f2 = nn.Conv1d(out_sz, 1, 1)

        self.bias = nn.Parameter(torch.zeros(out_sz))

        if self.residual and in_sz != out_sz:
            self.conv_residual = nn.Conv1d(in_sz, out_sz, 1)

        self.in_dropout = nn.Dropout(in_drop)
        self.coef_dropout = nn.Dropout(coef_drop)

    def forward(self, seq, bias_mat):
        if self.in_drop > 0.0 and self.training:
            seq = self.in_dropout(seq)

        seq_fts = self.conv_seq(seq.transpose(1, 2)).transpose(1, 2)

        f_1 = self.conv_f1(seq_fts.transpose(1, 2)).transpose(1, 2)
        f_2 = self.conv_f2(seq_fts.transpose(1, 2)).transpose(1, 2)
        logits = f_1 + f_2.transpose(1, 2)
        coefs = torch.softmax(F.leaky_relu(logits) + bias_mat, dim=-1)

        if self.coef_drop > 0.0 and self.training:
            coefs = self.coef_dropout(coefs)
        if self.in_drop > 0.0 and self.training:
            seq_fts = self.in_dropout(seq_fts)

        vals = torch.matmul(coefs, seq_fts)
        ret = vals + self.bias

        if self.residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + self.conv_residual(seq.transpose(1, 2)).transpose(1, 2)
            else:
                ret = ret + seq

        return self.activation(ret)


class ConstAttnHead(nn.Module):
    def __init__(self, in_sz, out_sz, activation=F.elu, in_drop=0.0, residual=False):
        super(ConstAttnHead, self).__init__()
        self.in_sz = in_sz
        self.out_sz = out_sz
        self.activation = activation
        self.in_drop = in_drop
        self.residual = residual

        self.conv_seq = nn.Conv1d(in_sz, out_sz, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_sz))

        if self.residual and in_sz != out_sz:
            self.conv_residual = nn.Conv1d(in_sz, out_sz, 1)

        self.in_dropout = nn.Dropout(in_drop)

    def forward(self, seq, adj_mat):
        if self.in_drop > 0.0 and self.training:
            seq = self.in_dropout(seq)

        seq_fts = self.conv_seq(seq.transpose(1, 2)).transpose(1, 2)

        deg = torch.sum(adj_mat, dim=-1, keepdim=True)
        deg = torch.clamp(deg, min=1.0)
        coefs = adj_mat / deg
        coefs = torch.where(torch.isnan(coefs), torch.zeros_like(coefs), coefs)

        vals = torch.matmul(coefs, seq_fts)
        ret = vals + self.bias

        if self.residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + self.conv_residual(seq.transpose(1, 2)).transpose(1, 2)
            else:
                ret = ret + seq

        return self.activation(ret)


class AvgHead(nn.Module):
    def __init__(self, in_sz, out_sz, activation=F.elu, in_drop=0.0, residual=False):
        super(AvgHead, self).__init__()
        self.in_sz = in_sz
        self.out_sz = out_sz
        self.activation = activation
        self.in_drop = in_drop
        self.residual = residual

        self.fc = nn.Linear(in_sz, out_sz, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_sz))

        if self.residual and in_sz != out_sz:
            self.fc_residual = nn.Linear(in_sz, out_sz)

        self.in_dropout = nn.Dropout(in_drop)

    def forward(self, seq, adj_mat):
        if self.in_drop > 0.0 and self.training:
            seq = self.in_dropout(seq)

        seq_fts = self.fc(seq)

        adj_with_self = adj_mat + torch.eye(adj_mat.shape[1], device=adj_mat.device).unsqueeze(0)
        deg = torch.sum(adj_with_self, dim=-1, keepdim=True)
        deg = torch.clamp(deg, min=1.0)
        coefs = adj_with_self / deg

        vals = torch.matmul(coefs, seq_fts)
        ret = vals + self.bias

        if self.residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + self.fc_residual(seq)
            else:
                ret = ret + seq

        return self.activation(ret)
