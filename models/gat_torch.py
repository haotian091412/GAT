import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.layers_torch import AttnHead, AvgHead


class GAT(nn.Module):
    def __init__(self, in_sz, nb_classes, nb_nodes, hid_units, n_heads, activation=F.elu, residual=False, 
                 attn_drop=0.0, ffd_drop=0.0):
        super(GAT, self).__init__()
        self.nb_classes = nb_classes
        self.nb_nodes = nb_nodes
        self.hid_units = hid_units
        self.n_heads = n_heads
        self.activation = activation
        self.residual = residual

        self.attn_layers = nn.ModuleList()
        self.attn_layers.append(nn.ModuleList([
            AttnHead(in_sz, hid_units[0], activation, ffd_drop, attn_drop, residual=False)
            for _ in range(n_heads[0])
        ]))

        for i in range(1, len(hid_units)):
            self.attn_layers.append(nn.ModuleList([
                AttnHead(hid_units[i-1] * n_heads[i-1], hid_units[i], activation, ffd_drop, attn_drop, residual)
                for _ in range(n_heads[i])
            ]))

        self.out_attns = nn.ModuleList([
            AttnHead(hid_units[-1] * n_heads[-2], nb_classes, lambda x: x, ffd_drop, attn_drop, residual=False)
            for _ in range(n_heads[-1])
        ])

    def forward(self, x, bias_mat):
        attns = []
        for attn in self.attn_layers[0]:
            attns.append(attn(x, bias_mat))
        h = torch.cat(attns, dim=-1)

        for i in range(1, len(self.attn_layers)):
            attns = []
            for attn in self.attn_layers[i]:
                attns.append(attn(h, bias_mat))
            h = torch.cat(attns, dim=-1)

        outs = []
        for out_attn in self.out_attns:
            outs.append(out_attn(h, bias_mat))
        logits = torch.stack(outs).mean(dim=0)

        return logits


class GATAvg(nn.Module):
    def __init__(self, in_sz, nb_classes, nb_nodes, hid_units, n_heads, activation=F.elu, residual=False, 
                 in_drop=0.0):
        super(GATAvg, self).__init__()
        self.nb_classes = nb_classes
        self.nb_nodes = nb_nodes
        self.hid_units = hid_units
        self.n_heads = n_heads
        self.activation = activation
        self.residual = residual

        self.avg_layers = nn.ModuleList()
        self.avg_layers.append(nn.ModuleList([
            AvgHead(in_sz, hid_units[0], activation, in_drop, residual=False)
            for _ in range(n_heads[0])
        ]))

        for i in range(1, len(hid_units)):
            self.avg_layers.append(nn.ModuleList([
                AvgHead(hid_units[i-1] * n_heads[i-1], hid_units[i], activation, in_drop, residual)
                for _ in range(n_heads[i])
            ]))

        self.out_avgs = nn.ModuleList([
            AvgHead(hid_units[-1] * n_heads[-2], nb_classes, lambda x: x, in_drop, residual=False)
            for _ in range(n_heads[-1])
        ])

    def forward(self, x, adj_mask):
        avgs = []
        for avg in self.avg_layers[0]:
            avgs.append(avg(x, adj_mask))
        h = torch.cat(avgs, dim=-1)

        for i in range(1, len(self.avg_layers)):
            avgs = []
            for avg in self.avg_layers[i]:
                avgs.append(avg(h, adj_mask))
            h = torch.cat(avgs, dim=-1)

        outs = []
        for out_avg in self.out_avgs:
            outs.append(out_avg(h, adj_mask))
        logits = torch.stack(outs).mean(dim=0)

        return logits
