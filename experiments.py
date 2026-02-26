"""
Experiments to investigate the impact of attention mechanism in GAT

Two methods to demonstrate attention influence:
1. Uniform Attention: Replace learned attention coefficients with uniform weights (1/degree)
2. Random Attention: Use random/fixed attention coefficients instead of learned ones
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from gat_pytorch import (
    AttentionHead, GAT, masked_softmax_cross_entropy, masked_accuracy,
    train_gat, load_and_prepare_data
)


class UniformAttentionHead(nn.Module):
    """
    Attention head with uniform attention weights
    This removes the learnable attention mechanism by using uniform weights
    """
    def __init__(self, in_features, out_features, dropout=0.0, alpha=0.2, residual=False):
        super(UniformAttentionHead, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.residual = residual

        # Only keep the linear transformation, remove attention parameters
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_features))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)

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

        # Create uniform attention coefficients
        # bias_mat has -1e9 for non-neighbors, 0 for neighbors
        # Create a mask for valid connections
        mask = (bias_mat > -1e8).float()  # 1 for neighbors, 0 for non-neighbors

        # Uniform weights: each node distributes attention uniformly to its neighbors
        # Sum of each row in mask gives the degree
        degree = mask.sum(dim=-1, keepdim=True)  # (batch_size, N, 1)
        degree = torch.clamp(degree, min=1.0)  # Avoid division by zero

        # Uniform attention coefficients
        coefs = mask / degree  # (batch_size, N, N)

        # Apply dropout to attention coefficients
        if self.dropout > 0:
            coefs = F.dropout(coefs, p=self.dropout, training=self.training)

        # Apply dropout to features
        h_drop = h
        if self.dropout > 0:
            h_drop = F.dropout(h_drop, p=self.dropout, training=self.training)

        # Aggregate features using uniform attention coefficients
        out = torch.matmul(coefs, h_drop)  # (batch_size, N, out_features)

        # Add bias
        out = out + self.bias

        # Residual connection
        if self.residual:
            if x.shape[-1] != out.shape[-1]:
                residual_proj = nn.Linear(x.shape[-1], out.shape[-1], bias=False).to(x.device)
                out = out + residual_proj(x)
            else:
                out = out + x

        return out


class RandomAttentionHead(nn.Module):
    """
    Attention head with random/fixed attention weights
    This uses randomly initialized but fixed attention coefficients
    """
    def __init__(self, in_features, out_features, dropout=0.0, alpha=0.2, residual=False):
        super(RandomAttentionHead, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.residual = residual

        # Only keep the linear transformation
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_features))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)

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

        # Create random attention coefficients (fixed during training)
        # Use the bias_mat structure to create a valid random attention
        mask = (bias_mat > -1e8).float()  # 1 for neighbors, 0 for non-neighbors

        # Generate random attention scores (fixed, not learnable)
        # We use a fixed seed based on the mask shape for reproducibility
        torch.manual_seed(42)
        random_scores = torch.randn_like(mask)  # Random scores
        random_scores = random_scores * mask + (1 - mask) * (-1e9)  # Mask non-neighbors

        # Apply softmax to get random attention coefficients
        coefs = F.softmax(random_scores, dim=-1)

        # Apply dropout to attention coefficients
        if self.dropout > 0:
            coefs = F.dropout(coefs, p=self.dropout, training=self.training)

        # Apply dropout to features
        h_drop = h
        if self.dropout > 0:
            h_drop = F.dropout(h_drop, p=self.dropout, training=self.training)

        # Aggregate features using random attention coefficients
        out = torch.matmul(coefs, h_drop)  # (batch_size, N, out_features)

        # Add bias
        out = out + self.bias

        # Residual connection
        if self.residual:
            if x.shape[-1] != out.shape[-1]:
                residual_proj = nn.Linear(x.shape[-1], out.shape[-1], bias=False).to(x.device)
                out = out + residual_proj(x)
            else:
                out = out + x

        return out


class GATVariant(nn.Module):
    """
    GAT with different attention mechanisms
    """
    def __init__(self, in_features, nb_classes, nb_nodes, hid_units, n_heads,
                 dropout=0.0, alpha=0.2, residual=False, attention_type='learned'):
        """
        attention_type: 'learned' (standard GAT), 'uniform', or 'random'
        """
        super(GATVariant, self).__init__()
        self.in_features = in_features
        self.nb_classes = nb_classes
        self.nb_nodes = nb_nodes
        self.hid_units = hid_units
        self.n_heads = n_heads
        self.dropout = dropout
        self.alpha = alpha
        self.residual = residual
        self.attention_type = attention_type

        # Select attention head type
        if attention_type == 'uniform':
            HeadClass = UniformAttentionHead
        elif attention_type == 'random':
            HeadClass = RandomAttentionHead
        else:
            HeadClass = AttentionHead

        # Build layers
        self.attention_layers = nn.ModuleList()

        # First layer (multi-head)
        first_layer = nn.ModuleList()
        for _ in range(n_heads[0]):
            first_layer.append(
                HeadClass(in_features, hid_units[0], dropout, alpha, residual=False)
            )
        self.attention_layers.append(first_layer)

        # Hidden layers
        for i in range(1, len(hid_units)):
            layer = nn.ModuleList()
            for _ in range(n_heads[i]):
                layer.append(
                    HeadClass(hid_units[i-1] * n_heads[i-1], hid_units[i], dropout, alpha, residual)
                )
            self.attention_layers.append(layer)

        # Output layer
        self.output_layer = nn.ModuleList()
        for _ in range(n_heads[-1]):
            self.output_layer.append(
                HeadClass(hid_units[-1] * n_heads[-2] if len(hid_units) > 1 else hid_units[-1] * n_heads[0],
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


def run_experiment(attention_type, seed=42):
    """
    Run experiment with specified attention type
    """
    print(f"\n{'='*60}")
    print(f"Running experiment: {attention_type.upper()} ATTENTION")
    print(f"{'='*60}")

    # Hyperparameters (same for all experiments)
    dataset = 'cora'
    hid_units = [8]
    n_heads = [8, 1]
    lr = 0.005
    l2_coef = 0.0005
    dropout = 0.6
    alpha = 0.2
    nb_epochs = 100000
    patience = 100

    print(f'Attention Type: {attention_type}')
    print(f'Seed: {seed}')

    # Load data
    (features, labels, train_mask, val_mask, test_mask, biases,
     nb_nodes, ft_size, nb_classes) = load_and_prepare_data(dataset)

    # Create model
    model = GATVariant(
        in_features=ft_size,
        nb_classes=nb_classes,
        nb_nodes=nb_nodes,
        hid_units=hid_units,
        n_heads=n_heads,
        dropout=dropout,
        alpha=alpha,
        residual=False,
        attention_type=attention_type
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
    save_path = f'pre_trained/cora_torch/{attention_type}_seed{seed}_best.pt'
    torch.save(model.state_dict(), save_path)
    print(f'Model saved to {save_path}')

    return history, test_acc


def plot_results(results, output_file='results_comparison.png'):
    """
    Plot training curves for all experiments
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = {
        'learned': '#1f77b4',  # Blue
        'uniform': '#ff7f0e',  # Orange
        'random': '#2ca02c'    # Green
    }

    labels = {
        'learned': 'GAT (Learned Attention)',
        'uniform': 'GAT (Uniform Attention)',
        'random': 'GAT (Random Attention)'
    }

    # Plot training loss
    ax = axes[0, 0]
    for exp_name, (history, _) in results.items():
        epochs = range(len(history['train_loss']))
        ax.plot(epochs, history['train_loss'], label=labels[exp_name],
                color=colors[exp_name], alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title('Training Loss Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot training accuracy
    ax = axes[0, 1]
    for exp_name, (history, _) in results.items():
        epochs = range(len(history['train_acc']))
        ax.plot(epochs, history['train_acc'], label=labels[exp_name],
                color=colors[exp_name], alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Accuracy')
    ax.set_title('Training Accuracy Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot validation loss
    ax = axes[1, 0]
    for exp_name, (history, _) in results.items():
        epochs = range(len(history['val_loss']))
        ax.plot(epochs, history['val_loss'], label=labels[exp_name],
                color=colors[exp_name], alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Validation Loss Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot validation accuracy
    ax = axes[1, 1]
    for exp_name, (history, _) in results.items():
        epochs = range(len(history['val_acc']))
        ax.plot(epochs, history['val_acc'], label=labels[exp_name],
                color=colors[exp_name], alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Accuracy')
    ax.set_title('Validation Accuracy Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {output_file}")
    plt.close()


def print_summary(results):
    """
    Print summary of all experiments
    """
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)

    print("\nTest Accuracies:")
    print("-" * 40)
    for exp_name, (_, test_acc) in results.items():
        print(f"{exp_name:15s}: {test_acc*100:.2f}%")

    print("\nFinal Training Loss:")
    print("-" * 40)
    for exp_name, (history, _) in results.items():
        final_loss = history['train_loss'][-1]
        print(f"{exp_name:15s}: {final_loss:.5f}")

    print("\nFinal Validation Accuracy:")
    print("-" * 40)
    for exp_name, (history, _) in results.items():
        final_acc = history['val_acc'][-1]
        print(f"{exp_name:15s}: {final_acc*100:.2f}%")

    # Calculate improvement
    if 'learned' in results and 'uniform' in results:
        learned_acc = results['learned'][1]
        uniform_acc = results['uniform'][1]
        improvement = (learned_acc - uniform_acc) / uniform_acc * 100
        print(f"\nLearned Attention vs Uniform Attention:")
        print(f"  Relative improvement: {improvement:+.2f}%")

    print("="*80)


if __name__ == '__main__':
    # Set random seeds for reproducibility
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Run all experiments
    results = {}

    # 1. Baseline: Standard GAT with learned attention
    history_learned, test_acc_learned = run_experiment('learned', seed=SEED)
    results['learned'] = (history_learned, test_acc_learned)

    # 2. Experiment 1: Uniform attention (no learnable attention)
    history_uniform, test_acc_uniform = run_experiment('uniform', seed=SEED)
    results['uniform'] = (history_uniform, test_acc_uniform)

    # 3. Experiment 2: Random attention (fixed random attention coefficients)
    history_random, test_acc_random = run_experiment('random', seed=SEED)
    results['random'] = (history_random, test_acc_random)

    # Print summary
    print_summary(results)

    # Plot results
    plot_results(results, output_file='results_comparison_final.png')
