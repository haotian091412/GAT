import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.layers_pytorch import AttnHead, NoAttnHead, SingleHeadAttn

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, n_heads, dropout=0.6, alpha=0.2):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.n_heads = n_heads
        
        self.attentions = nn.ModuleList()
        for _ in range(n_heads[0]):
            self.attentions.append(AttnHead(nfeat, nhid[0], dropout=dropout, alpha=alpha, concat=True))
        
        if len(nhid) > 1:
            self.attentions2 = nn.ModuleList()
            for _ in range(n_heads[1]):
                self.attentions2.append(AttnHead(nhid[0] * n_heads[0], nhid[1], dropout=dropout, alpha=alpha, concat=True))
        
        self.out_att = nn.ModuleList()
        for _ in range(n_heads[-1]):
            if len(nhid) > 1:
                self.out_att.append(AttnHead(nhid[1] * n_heads[1], nclass, dropout=dropout, alpha=alpha, concat=False))
            else:
                self.out_att.append(AttnHead(nhid[0] * n_heads[0], nclass, dropout=dropout, alpha=alpha, concat=False))
    
    def forward(self, x, adj_bias):
        x = F.dropout(x, self.dropout, training=self.training)
        
        x = torch.cat([att(x, adj_bias) for att in self.attentions], dim=2)
        
        if hasattr(self, 'attentions2'):
            x = F.dropout(x, self.dropout, training=self.training)
            x = torch.cat([att(x, adj_bias) for att in self.attentions2], dim=2)
        
        x = F.dropout(x, self.dropout, training=self.training)
        out = torch.stack([att(x, adj_bias) for att in self.out_att], dim=0)
        out = out.mean(dim=0)
        
        return out


class NoAttnGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, n_heads, dropout=0.6):
        super(NoAttnGAT, self).__init__()
        self.dropout = dropout
        self.n_heads = n_heads
        
        self.attentions = nn.ModuleList()
        for _ in range(n_heads[0]):
            self.attentions.append(NoAttnHead(nfeat, nhid[0], dropout=dropout, concat=True))
        
        if len(nhid) > 1:
            self.attentions2 = nn.ModuleList()
            for _ in range(n_heads[1]):
                self.attentions2.append(NoAttnHead(nhid[0] * n_heads[0], nhid[1], dropout=dropout, concat=True))
        
        self.out_att = nn.ModuleList()
        for _ in range(n_heads[-1]):
            if len(nhid) > 1:
                self.out_att.append(NoAttnHead(nhid[1] * n_heads[1], nclass, dropout=dropout, concat=False))
            else:
                self.out_att.append(NoAttnHead(nhid[0] * n_heads[0], nclass, dropout=dropout, concat=False))
    
    def forward(self, x, adj_bias):
        x = F.dropout(x, self.dropout, training=self.training)
        
        x = torch.cat([att(x, adj_bias) for att in self.attentions], dim=2)
        
        if hasattr(self, 'attentions2'):
            x = F.dropout(x, self.dropout, training=self.training)
            x = torch.cat([att(x, adj_bias) for att in self.attentions2], dim=2)
        
        x = F.dropout(x, self.dropout, training=self.training)
        out = torch.stack([att(x, adj_bias) for att in self.out_att], dim=0)
        out = out.mean(dim=0)
        
        return out


class SingleHeadGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.6, alpha=0.2):
        super(SingleHeadGAT, self).__init__()
        self.dropout = dropout
        
        self.attention1 = SingleHeadAttn(nfeat, nhid[0], dropout=dropout, alpha=alpha, concat=True)
        
        if len(nhid) > 1:
            self.attention2 = SingleHeadAttn(nhid[0], nhid[1], dropout=dropout, alpha=alpha, concat=True)
            self.out_attention = SingleHeadAttn(nhid[1], nclass, dropout=dropout, alpha=alpha, concat=False)
        else:
            self.out_attention = SingleHeadAttn(nhid[0], nclass, dropout=dropout, alpha=alpha, concat=False)
    
    def forward(self, x, adj_bias):
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.attention1(x, adj_bias)
        
        if hasattr(self, 'attention2'):
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.attention2(x, adj_bias)
        
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_attention(x, adj_bias)
        
        return x
