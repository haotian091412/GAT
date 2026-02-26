import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import importlib.util

spec = importlib.util.spec_from_file_location("layers_torch", "utils/layers_torch.py")
layers_torch = importlib.util.module_from_spec(spec)
sys.modules["layers_torch"] = layers_torch
spec.loader.exec_module(layers_torch)
AttnHead = layers_torch.AttnHead


class GATTorch(nn.Module):
    def __init__(self, in_sz, nb_classes, nb_nodes, hid_units, n_heads, activation=F.elu, residual=False,
                 attn_drop=0.0, ffd_drop=0.0):
        super(GATTorch, self).__init__()
        self.in_sz = in_sz
        self.nb_classes = nb_classes
        self.nb_nodes = nb_nodes
        self.hid_units = hid_units
        self.n_heads = n_heads
        self.activation = activation
        self.residual = residual
        self.attn_drop = attn_drop
        self.ffd_drop = ffd_drop

        self.layers = nn.ModuleList()

        prev_sz = in_sz
        for i in range(len(hid_units)):
            layer_heads = nn.ModuleList()
            for _ in range(n_heads[i]):
                layer_heads.append(
                    AttnHead(prev_sz, hid_units[i], activation=activation,
                             in_drop=ffd_drop, coef_drop=attn_drop, residual=residual if i > 0 else False)
                )
            self.layers.append(layer_heads)
            prev_sz = hid_units[i] * n_heads[i]

        self.out_heads = nn.ModuleList()
        for _ in range(n_heads[-1]):
            self.out_heads.append(
                AttnHead(prev_sz, nb_classes, activation=lambda x: x,
                         in_drop=ffd_drop, coef_drop=attn_drop, residual=False)
            )

    def forward(self, inputs, bias_mat):
        x = inputs

        for i, layer_heads in enumerate(self.layers):
            head_outputs = []
            for head in layer_heads:
                head_outputs.append(head(x, bias_mat))
            x = torch.cat(head_outputs, dim=-1)

        out = []
        for head in self.out_heads:
            out.append(head(x, bias_mat))
        logits = torch.stack(out, dim=0).mean(dim=0)

        return logits
