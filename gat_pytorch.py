"""
PyTorch implementation of Graph Attention Network (GAT)
Converted from TensorFlow implementation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.process import load_data, preprocess_features, adj_to_bias


class AttentionHead(nn.Module):
    """
    Single attention head for GAT
    """
    def __init__(self, in_features, out_features, dropout=0.0, alpha=0.2, residual=False):
        super(AttentionHead, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.residual = residual

        # Linear transformation for features
        self.W = nn.Linear(in_features, out_features, bias=False)

        # Attention parameters (f_1 and f_2 in the paper)
        self.a_1 = nn.Linear(out_features, 1, bias=False)
        self.a_2 = nn.Linear(out_features, 1, bias=False)

        # Bias term
        self.bias = nn.Parameter(torch.zeros(out_features))

        # LeakyReLU activation
        self.leakyrelu = nn.LeakyReLU(self.alpha)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_1.weight)
        nn.init.xavier_uniform_(self.a_2.weight)

    def forward(self, x, bias_mat):
        """
        x: (batch_size, N, in_features)
        bias_mat: (batch_size, N, N) - contains -1e9 for non-neighbor positions
        """
        # Apply dropout to input
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Linear transformation
        h = self.W(x)  # (batch_size, N, out_features)

        # Compute attention coefficients
        f_1 = self.a_1(h)  # (batch_size, N, 1)
        f_2 = self.a_2(h)  # (batch_size, N, 1)

        # Compute attention scores (e = f_1 + f_2^T)
        logits = f_1 + f_2.transpose(1, 2)  # (batch_size, N, N)

        # Apply LeakyReLU and add bias (for masking non-neighbors)
        logits = self.leakyrelu(logits) + bias_mat

        # Apply softmax to get attention coefficients
        coefs = F.softmax(logits, dim=-1)

        # Apply dropout to attention coefficients
        if self.dropout > 0:
            coefs = F.dropout(coefs, p=self.dropout, training=self.training)

        # Apply dropout to features
        h_drop = h
        if self.dropout > 0:
            h_drop = F.dropout(h_drop, p=self.dropout, training=self.training)

        # Aggregate features using attention coefficients
        out = torch.matmul(coefs, h_drop)  # (batch_size, N, out_features)

        # Add bias
        out = out + self.bias

        # Residual connection
        if self.residual:
            if x.shape[-1] != out.shape[-1]:
                # Project input to match output dimension
                residual_proj = nn.Linear(x.shape[-1], out.shape[-1], bias=False).to(x.device)
                out = out + residual_proj(x)
            else:
                out = out + x

        return out


class GAT(nn.Module):
    """
    Graph Attention Network
    """
    def __init__(self, in_features, nb_classes, nb_nodes, hid_units, n_heads,
                 dropout=0.0, alpha=0.2, residual=False):
        super(GAT, self).__init__()
        self.in_features = in_features
        self.nb_classes = nb_classes
        self.nb_nodes = nb_nodes
        self.hid_units = hid_units
        self.n_heads = n_heads
        self.dropout = dropout
        self.alpha = alpha
        self.residual = residual

        # Build layers
        self.attention_layers = nn.ModuleList()

        # First layer (multi-head)
        first_layer = nn.ModuleList()
        for _ in range(n_heads[0]):
            first_layer.append(
                AttentionHead(in_features, hid_units[0], dropout, alpha, residual=False)
            )
        self.attention_layers.append(first_layer)

        # Hidden layers
        for i in range(1, len(hid_units)):
            layer = nn.ModuleList()
            for _ in range(n_heads[i]):
                layer.append(
                    AttentionHead(hid_units[i-1] * n_heads[i-1], hid_units[i], dropout, alpha, residual)
                )
            self.attention_layers.append(layer)

        # Output layer
        self.output_layer = nn.ModuleList()
        for _ in range(n_heads[-1]):
            self.output_layer.append(
                AttentionHead(hid_units[-1] * n_heads[-2] if len(hid_units) > 1 else hid_units[-1] * n_heads[0],
                             nb_classes, dropout, alpha, residual=False)
            )

    def forward(self, x, bias_mat):
        """
        x: (batch_size, N, in_features)
        bias_mat: (batch_size, N, N)
        """
        # First layer
        h = x
        attn_outputs = []
        for attn_head in self.attention_layers[0]:
            attn_outputs.append(attn_head(h, bias_mat))
        h = torch.cat(attn_outputs, dim=-1)
        h = F.elu(h)

        # Hidden layers
        for layer_idx in range(1, len(self.hid_units)):
            h_old = h
            attn_outputs = []
            for attn_head in self.attention_layers[layer_idx]:
                attn_outputs.append(attn_head(h, bias_mat))
            h = torch.cat(attn_outputs, dim=-1)
            h = F.elu(h)

        # Output layer
        out_outputs = []
        for attn_head in self.output_layer:
            out_outputs.append(attn_head(h, bias_mat))

        # Average output heads
        logits = torch.stack(out_outputs, dim=0).mean(dim=0)

        return logits


def masked_softmax_cross_entropy(logits, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = F.cross_entropy(logits, labels, reduction='none')
    mask = mask.float()
    mask = mask / mask.mean()
    loss = loss * mask
    return loss.mean()


def masked_accuracy(logits, labels, mask):
    """Accuracy with masking."""
    preds = torch.argmax(logits, dim=1)
    correct = (preds == labels).float()
    mask = mask.float()
    mask = mask / mask.mean()
    correct = correct * mask
    return correct.mean()


def train_gat(model, features, labels, train_mask, val_mask, test_mask, bias_mat,
              lr=0.005, l2_coef=0.0005, nb_epochs=100000, patience=100,
              device='cpu', seed=42):
    """
    Train GAT model
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = model.to(device)
    features = features.to(device)
    labels = labels.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)
    bias_mat = bias_mat.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)

    best_val_acc = 0.0
    best_val_loss = float('inf')
    curr_step = 0
    best_model_state = None

    # History for plotting
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    for epoch in range(nb_epochs):
        model.train()
        optimizer.zero_grad()

        logits = model(features, bias_mat)
        logits_reshaped = logits.reshape(-1, logits.shape[-1])
        labels_reshaped = labels.reshape(-1)
        train_mask_reshaped = train_mask.reshape(-1)

        loss = masked_softmax_cross_entropy(logits_reshaped, labels_reshaped, train_mask_reshaped)
        acc = masked_accuracy(logits_reshaped, labels_reshaped, train_mask_reshaped)

        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(features, bias_mat)
            val_logits_reshaped = val_logits.reshape(-1, val_logits.shape[-1])
            val_mask_reshaped = val_mask.reshape(-1)
            labels_reshaped = labels.reshape(-1)

            val_loss = masked_softmax_cross_entropy(val_logits_reshaped, labels_reshaped, val_mask_reshaped)
            val_acc = masked_accuracy(val_logits_reshaped, labels_reshaped, val_mask_reshaped)

        # Record history
        history['train_loss'].append(loss.item())
        history['train_acc'].append(acc.item())
        history['val_loss'].append(val_loss.item())
        history['val_acc'].append(val_acc.item())

        if epoch % 50 == 0:
            print(f'Epoch {epoch}: Train loss = {loss.item():.5f}, acc = {acc.item():.5f} | '
                  f'Val loss = {val_loss.item():.5f}, acc = {val_acc.item():.5f}')

        # Early stopping
        if val_acc >= best_val_acc or val_loss <= best_val_loss:
            if val_acc >= best_val_acc and val_loss <= best_val_loss:
                best_model_state = model.state_dict().copy()
            best_val_acc = max(val_acc.item(), best_val_acc)
            best_val_loss = min(val_loss.item(), best_val_loss)
            curr_step = 0
        else:
            curr_step += 1
            if curr_step == patience:
                print(f'Early stop! Min loss: {best_val_loss:.5f}, Max accuracy: {best_val_acc:.5f}')
                break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Test
    model.eval()
    with torch.no_grad():
        test_logits = model(features, bias_mat)
        test_logits_reshaped = test_logits.reshape(-1, test_logits.shape[-1])
        test_mask_reshaped = test_mask.reshape(-1)
        labels_reshaped = labels.reshape(-1)

        test_loss = masked_softmax_cross_entropy(test_logits_reshaped, labels_reshaped, test_mask_reshaped)
        test_acc = masked_accuracy(test_logits_reshaped, labels_reshaped, test_mask_reshaped)

    print(f'Test loss: {test_loss.item():.5f}, Test accuracy: {test_acc.item():.5f}')

    return model, history, test_acc.item()


def load_and_prepare_data(dataset='cora'):
    """Load and prepare data for training"""
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(dataset)
    features, _ = preprocess_features(features)

    nb_nodes = features.shape[0]
    ft_size = features.shape[1]
    nb_classes = y_train.shape[1]

    adj = adj.todense()

    # Add batch dimension
    features = features[np.newaxis]
    adj = adj[np.newaxis]
    y_train = y_train[np.newaxis]
    y_val = y_val[np.newaxis]
    y_test = y_test[np.newaxis]
    train_mask = train_mask[np.newaxis]
    val_mask = val_mask[np.newaxis]
    test_mask = test_mask[np.newaxis]

    biases = adj_to_bias(adj, [nb_nodes], nhood=1)

    # Convert to torch tensors
    features = torch.FloatTensor(features)
    labels = torch.LongTensor(np.argmax(y_train + y_val + y_test, axis=-1))
    train_mask = torch.BoolTensor(train_mask)
    val_mask = torch.BoolTensor(val_mask)
    test_mask = torch.BoolTensor(test_mask)
    biases = torch.FloatTensor(biases)

    return (features, labels, train_mask, val_mask, test_mask, biases,
            nb_nodes, ft_size, nb_classes)


if __name__ == '__main__':
    # Hyperparameters (same as original TensorFlow implementation)
    dataset = 'cora'
    hid_units = [8]  # numbers of hidden units per each attention head in each layer
    n_heads = [8, 1]  # additional entry for the output layer
    lr = 0.005
    l2_coef = 0.0005
    dropout = 0.6
    alpha = 0.2
    nb_epochs = 100000
    patience = 100
    seed = 42

    print('Dataset:', dataset)
    print('----- Opt. hyperparams -----')
    print('lr:', lr)
    print('l2_coef:', l2_coef)
    print('dropout:', dropout)
    print('----- Archi. hyperparams -----')
    print('nb. layers:', len(hid_units))
    print('nb. units per layer:', hid_units)
    print('nb. attention heads:', n_heads)
    print('seed:', seed)

    # Load data
    (features, labels, train_mask, val_mask, test_mask, biases,
     nb_nodes, ft_size, nb_classes) = load_and_prepare_data(dataset)

    # Create model
    model = GAT(
        in_features=ft_size,
        nb_classes=nb_classes,
        nb_nodes=nb_nodes,
        hid_units=hid_units,
        n_heads=n_heads,
        dropout=dropout,
        alpha=alpha,
        residual=False
    )

    # Train
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    model, history, test_acc = train_gat(
        model, features, labels, train_mask, val_mask, test_mask, biases,
        lr=lr, l2_coef=l2_coef, nb_epochs=nb_epochs, patience=patience,
        device=device, seed=seed
    )

    # Save model
    torch.save(model.state_dict(), 'pre_trained/cora_torch/gat_baseline.pt')
    print(f'Model saved. Test accuracy: {test_acc:.5f}')
