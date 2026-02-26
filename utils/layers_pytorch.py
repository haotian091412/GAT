import torch
import torch.nn as nn
import torch.nn.functional as F

class AttnHead(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.0, alpha=0.2, concat=True):
        super(AttnHead, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a1 = nn.Parameter(torch.zeros(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a1.data, gain=1.414)
        self.a2 = nn.Parameter(torch.zeros(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a2.data, gain=1.414)
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        
    def forward(self, h, adj_bias):
        Wh = torch.matmul(h, self.W)
        
        f_1 = torch.matmul(Wh, self.a1)
        f_2 = torch.matmul(Wh, self.a2)
        
        logits = f_1 + f_2.transpose(1, 2)
        
        coefs = F.softmax(self.leakyrelu(logits) + adj_bias, dim=-1)
        
        coefs = F.dropout(coefs, self.dropout, training=self.training)
        Wh = F.dropout(Wh, self.dropout, training=self.training)
        
        h_prime = torch.matmul(coefs, Wh)
        
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


class NoAttnHead(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.0, concat=True):
        super(NoAttnHead, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.concat = concat
        
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
    def forward(self, h, adj_bias):
        Wh = torch.matmul(h, self.W)
        
        adj_normalized = adj_bias.clone()
        adj_normalized = torch.where(adj_normalized > -1e8, 
                                     torch.ones_like(adj_normalized), 
                                     torch.zeros_like(adj_normalized))
        
        degree = adj_normalized.sum(dim=-1, keepdim=True)
        degree = torch.where(degree > 0, degree, torch.ones_like(degree))
        adj_normalized = adj_normalized / degree
        
        adj_normalized = F.dropout(adj_normalized, self.dropout, training=self.training)
        Wh = F.dropout(Wh, self.dropout, training=self.training)
        
        h_prime = torch.matmul(adj_normalized, Wh)
        
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


class SingleHeadAttn(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.0, alpha=0.2, concat=True):
        super(SingleHeadAttn, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        
    def forward(self, h, adj_bias):
        Wh = torch.matmul(h, self.W)
        
        N = h.size()[1]
        
        a_input = torch.cat([
            Wh.repeat(1, 1, N).view(h.size()[0], N * N, self.out_features),
            Wh.repeat(1, N, 1)
        ], dim=2).view(h.size()[0], N, N, 2 * self.out_features)
        
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(-1))
        
        attention = F.softmax(e + adj_bias, dim=-1)
        
        attention = F.dropout(attention, self.dropout, training=self.training)
        Wh = F.dropout(Wh, self.dropout, training=self.training)
        
        h_prime = torch.matmul(attention, Wh)
        
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime
