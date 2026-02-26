"""
Attention Visualization for GAT

This script visualizes the attention weights learned by GAT to verify:
1. Whether attention weights correlate with node label similarity
2. Whether attention focuses on semantically similar neighbors
3. The distribution of attention weights across different edge types
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import networkx as nx
from sklearn.manifold import TSNE
from gat_pytorch import GAT, load_and_prepare_data
from utils.process import load_data, preprocess_features, adj_to_bias


class GATWithAttentionCapture(GAT):
    """
    GAT model that captures attention weights during forward pass
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attention_weights = []  # Store attention weights

    def forward(self, x, bias_mat, capture_attention=True):
        """
        Forward pass with attention capture
        """
        self.attention_weights = []  # Clear previous weights
        h = x

        # First layer
        attn_outputs = []
        layer_attentions = []
        for attn_head in self.attention_layers[0]:
            out, attn_weights = self._forward_head_with_attention(
                attn_head, h, bias_mat
            )
            attn_outputs.append(out)
            if capture_attention:
                layer_attentions.append(attn_weights)

        if capture_attention:
            self.attention_weights.append(torch.stack(layer_attentions, dim=0))

        h = torch.cat(attn_outputs, dim=-1)
        h = F.elu(h)

        # Hidden layers
        for layer_idx in range(1, len(self.hid_units)):
            attn_outputs = []
            layer_attentions = []
            for attn_head in self.attention_layers[layer_idx]:
                out, attn_weights = self._forward_head_with_attention(
                    attn_head, h, bias_mat
                )
                attn_outputs.append(out)
                if capture_attention:
                    layer_attentions.append(attn_weights)

            if capture_attention:
                self.attention_weights.append(torch.stack(layer_attentions, dim=0))

            h = torch.cat(attn_outputs, dim=-1)
            h = F.elu(h)

        # Output layer
        out_outputs = []
        for attn_head in self.output_layer:
            out, _ = self._forward_head_with_attention(attn_head, h, bias_mat)
            out_outputs.append(out)

        logits = torch.stack(out_outputs, dim=0).mean(dim=0)

        return logits

    def _forward_head_with_attention(self, attn_head, x, bias_mat):
        """
        Forward pass for a single attention head, returning both output and attention weights
        """
        if self.training and self.dropout > 0:
            x_in = F.dropout(x, p=self.dropout, training=self.training)
        else:
            x_in = x

        h = attn_head.W(x_in)

        # Compute attention coefficients
        f_1 = attn_head.a_1(h)
        f_2 = attn_head.a_2(h)
        logits = f_1 + f_2.transpose(1, 2)
        logits = attn_head.leakyrelu(logits) + bias_mat

        # Softmax to get attention coefficients
        coefs = F.softmax(logits, dim=-1)

        # Apply dropout
        if self.training and self.dropout > 0:
            coefs = F.dropout(coefs, p=self.dropout, training=self.training)
            h = F.dropout(h, p=self.dropout, training=self.training)

        # Aggregate
        out = torch.matmul(coefs, h)
        out = out + attn_head.bias

        # Residual
        if attn_head.residual:
            if x.shape[-1] != out.shape[-1]:
                residual_proj = nn.Linear(x.shape[-1], out.shape[-1], bias=False).to(x.device)
                out = out + residual_proj(x)
            else:
                out = out + x

        return out, coefs


def analyze_attention_patterns(model, features, labels, train_mask, adj, bias_mat, device='cpu'):
    """
    Analyze attention patterns across the graph
    """
    model.eval()
    with torch.no_grad():
        features = features.to(device)
        bias_mat = bias_mat.to(device)
        logits = model(features, bias_mat, capture_attention=True)

    # Get attention weights from first layer (most interpretable)
    # Shape: (n_heads, batch_size, N, N)
    attn_weights = model.attention_weights[0].squeeze(1).cpu().numpy()  # (n_heads, N, N)

    # Average across heads
    avg_attn = attn_weights.mean(axis=0)  # (N, N)

    # Get predictions
    preds = torch.argmax(logits, dim=-1).squeeze().cpu().numpy()
    labels_np = labels.squeeze().cpu().numpy()
    train_mask_np = train_mask.squeeze().cpu().numpy()

    # Convert adjacency matrix
    adj_np = adj.squeeze().cpu().numpy() if torch.is_tensor(adj) else adj

    return avg_attn, preds, labels_np, train_mask_np, adj_np


def visualize_attention_heatmap(attn_matrix, labels, sample_size=100, save_path='attention_heatmap.png'):
    """
    Visualize attention weights as a heatmap, sorted by node labels
    """
    # Sample nodes for visualization
    n_nodes = attn_matrix.shape[0]
    if n_nodes > sample_size:
        # Stratified sampling to maintain class distribution
        indices = []
        for label in np.unique(labels):
            label_indices = np.where(labels == label)[0]
            n_sample = max(1, int(sample_size * len(label_indices) / n_nodes))
            sampled = np.random.choice(label_indices, min(n_sample, len(label_indices)), replace=False)
            indices.extend(sampled)
        indices = np.array(indices)
    else:
        indices = np.arange(n_nodes)

    # Sort by labels
    sorted_indices = indices[np.argsort(labels[indices])]

    # Extract submatrix
    sub_attn = attn_matrix[np.ix_(sorted_indices, sorted_indices)]
    sub_labels = labels[sorted_indices]

    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))

    # Create heatmap
    im = ax.imshow(sub_attn, cmap='viridis', aspect='auto')

    # Add colorbar
    plt.colorbar(im, ax=ax, label='Attention Weight')

    # Draw lines to separate classes
    unique_labels = np.unique(sub_labels)
    boundaries = []
    for i, label in enumerate(unique_labels[:-1]):
        boundary = np.where(sub_labels == label)[0][-1] + 0.5
        boundaries.append(boundary)
        ax.axhline(y=boundary, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax.axvline(x=boundary, color='red', linestyle='--', linewidth=1, alpha=0.5)

    ax.set_xlabel('Node Index (sorted by label)')
    ax.set_ylabel('Node Index (sorted by label)')
    ax.set_title('Attention Weight Heatmap (Nodes Sorted by Label)')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Attention heatmap saved to {save_path}")


def analyze_attention_by_edge_type(attn_matrix, adj, labels, save_path='attention_by_edge_type.png'):
    """
    Analyze attention distribution based on edge types (same class vs different class)
    """
    n_nodes = attn_matrix.shape[0]

    # Get edges from adjacency matrix
    edges = []
    same_class_attentions = []
    diff_class_attentions = []
    self_loop_attentions = []

    for i in range(n_nodes):
        for j in range(n_nodes):
            if adj[i, j] > 0:  # There is an edge
                if i == j:
                    self_loop_attentions.append(attn_matrix[i, j])
                elif labels[i] == labels[j]:
                    same_class_attentions.append(attn_matrix[i, j])
                else:
                    diff_class_attentions.append(attn_matrix[i, j])

    # Convert to numpy arrays
    same_class_attentions = np.array(same_class_attentions)
    diff_class_attentions = np.array(diff_class_attentions)
    self_loop_attentions = np.array(self_loop_attentions)

    # Plot distribution
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Histogram comparison
    bins = np.linspace(0, max(np.max(same_class_attentions) if len(same_class_attentions) > 0 else 0,
                               np.max(diff_class_attentions) if len(diff_class_attentions) > 0 else 0), 50)

    axes[0].hist(same_class_attentions, bins=bins, alpha=0.6, label='Same Class', color='green', density=True)
    axes[0].hist(diff_class_attentions, bins=bins, alpha=0.6, label='Different Class', color='red', density=True)
    axes[0].set_xlabel('Attention Weight')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Attention Distribution by Edge Type')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Box plot
    data_to_plot = [same_class_attentions, diff_class_attentions, self_loop_attentions]
    labels_plot = ['Same Class', 'Different Class', 'Self Loop']
    axes[1].boxplot(data_to_plot, labels=labels_plot)
    axes[1].set_ylabel('Attention Weight')
    axes[1].set_title('Attention Weight Distribution')
    axes[1].grid(True, alpha=0.3)

    # Statistics
    stats_text = f"""
    Attention Statistics:

    Same Class Edges:
      Mean: {np.mean(same_class_attentions):.4f}
      Std:  {np.std(same_class_attentions):.4f}
      Count: {len(same_class_attentions)}

    Different Class Edges:
      Mean: {np.mean(diff_class_attentions):.4f}
      Std:  {np.std(diff_class_attentions):.4f}
      Count: {len(diff_class_attentions)}

    Self Loops:
      Mean: {np.mean(self_loop_attentions):.4f}
      Std:  {np.std(self_loop_attentions):.4f}
      Count: {len(self_loop_attentions)}

    Same vs Diff (t-test):
      p-value: {ttest_ind_safe(same_class_attentions, diff_class_attentions)}
    """

    axes[2].text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                verticalalignment='center')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Attention by edge type saved to {save_path}")

    return {
        'same_class_mean': np.mean(same_class_attentions),
        'diff_class_mean': np.mean(diff_class_attentions),
        'self_loop_mean': np.mean(self_loop_attentions) if len(self_loop_attentions) > 0 else 0,
        'same_class_std': np.std(same_class_attentions),
        'diff_class_std': np.std(diff_class_attentions),
        'self_loop_std': np.std(self_loop_attentions) if len(self_loop_attentions) > 0 else 0,
    }


def ttest_ind_safe(a, b):
    """Safe t-test that handles edge cases"""
    from scipy import stats
    if len(a) < 2 or len(b) < 2:
        return "N/A (insufficient data)"
    try:
        _, pvalue = stats.ttest_ind(a, b)
        return f"{pvalue:.6f}"
    except Exception as e:
        return f"Error: {str(e)}"


def visualize_attention_graph(attn_matrix, adj, labels, node_features,
                              sample_size=50, save_path='attention_graph.png'):
    """
    Visualize the graph with attention-weighted edges
    """
    # Sample nodes
    n_nodes = attn_matrix.shape[0]
    if n_nodes > sample_size:
        indices = np.random.choice(n_nodes, sample_size, replace=False)
    else:
        indices = np.arange(n_nodes)

    # Create subgraph
    sub_attn = attn_matrix[np.ix_(indices, indices)]
    sub_adj = adj[np.ix_(indices, indices)]
    sub_labels = labels[indices]

    # Create networkx graph
    G = nx.Graph()

    # Add nodes with labels
    for i, idx in enumerate(indices):
        G.add_node(i, label=sub_labels[i])

    # Add edges with attention weights
    edge_weights = []
    for i in range(len(indices)):
        for j in range(i+1, len(indices)):
            if sub_adj[i, j] > 0:
                weight = sub_attn[i, j] + sub_attn[j, i]  # Sum both directions
                G.add_edge(i, j, weight=weight)
                edge_weights.append(weight)

    # Plot
    fig, ax = plt.subplots(figsize=(14, 12))

    # Layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    # Color nodes by label
    unique_labels = np.unique(sub_labels)
    colors = plt.cm.tab10(sub_labels / max(sub_labels))

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=300, alpha=0.8, ax=ax)

    # Draw edges with width proportional to attention
    if len(edge_weights) > 0:
        max_weight = max(edge_weights)
        min_weight = min(edge_weights)
        normalized_weights = [(w - min_weight) / (max_weight - min_weight + 1e-8) * 3 + 0.5
                             for w in edge_weights]

        edges = G.edges()
        nx.draw_networkx_edges(G, pos, width=normalized_weights, alpha=0.5,
                              edge_color='gray', ax=ax)

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)

    ax.set_title('Graph Visualization with Attention-Weighted Edges\n(Edge width = Attention weight)')
    ax.axis('off')

    # Add legend for classes
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor=plt.cm.tab10(i / max(unique_labels)),
                                  markersize=10, label=f'Class {i}')
                      for i in unique_labels]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Attention graph visualization saved to {save_path}")


def analyze_top_attention_neighbors(attn_matrix, labels, k=5, save_path='top_neighbors_analysis.png'):
    """
    Analyze the labels of top-k attention neighbors for each node
    """
    n_nodes = attn_matrix.shape[0]
    n_classes = len(np.unique(labels))

    # For each node, find top-k neighbors by attention
    same_class_ratios = []

    for i in range(n_nodes):
        # Get attention scores for node i
        attn_scores = attn_matrix[i].copy()
        attn_scores[i] = -1  # Exclude self

        # Find top-k neighbors
        top_k_indices = np.argsort(attn_scores)[-k:]

        # Count same class neighbors
        same_class_count = sum(labels[top_k_indices] == labels[i])
        same_class_ratio = same_class_count / k
        same_class_ratios.append(same_class_ratio)

    same_class_ratios = np.array(same_class_ratios)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram of same-class ratios
    axes[0].hist(same_class_ratios, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
    axes[0].axvline(x=np.mean(same_class_ratios), color='red', linestyle='--',
                   label=f'Mean: {np.mean(same_class_ratios):.3f}')
    axes[0].set_xlabel(f'Ratio of Same-Class in Top-{k} Neighbors')
    axes[0].set_ylabel('Number of Nodes')
    axes[0].set_title(f'Distribution of Same-Class Ratio in Top-{k} Attention Neighbors')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Same-class ratio by node class
    class_ratios = []
    class_names = []
    for c in range(n_classes):
        mask = labels == c
        if sum(mask) > 0:
            class_ratios.append(same_class_ratios[mask])
            class_names.append(f'Class {c}')

    axes[1].boxplot(class_ratios, labels=class_names)
    axes[1].set_ylabel(f'Same-Class Ratio in Top-{k} Neighbors')
    axes[1].set_xlabel('Node Class')
    axes[1].set_title(f'Same-Class Attention Ratio by Node Class')
    axes[1].grid(True, alpha=0.3)
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Top neighbors analysis saved to {save_path}")

    return {
        'mean_same_class_ratio': np.mean(same_class_ratios),
        'std_same_class_ratio': np.std(same_class_ratios),
        'same_class_ratios_by_class': {i: np.mean(ratios) for i, ratios in enumerate(class_ratios)}
    }


def compare_learned_vs_uniform_attention(model, features, labels, adj, bias_mat, device='cpu'):
    """
    Compare learned attention with uniform attention
    """
    model.eval()

    with torch.no_grad():
        features = features.to(device)
        bias_mat = bias_mat.to(device)

        # Get learned attention
        logits_learned = model(features, bias_mat, capture_attention=True)
        learned_attn = model.attention_weights[0].squeeze(1).mean(0).cpu().numpy()

    # Compute uniform attention (1/degree)
    degree = adj.sum(axis=1)
    degree = np.maximum(degree, 1)  # Avoid division by zero
    uniform_attn = adj / degree[:, np.newaxis]

    # Compare predictions
    preds_learned = torch.argmax(logits_learned, dim=-1).squeeze().cpu().numpy()

    # Compute accuracy for both
    labels_np = labels.squeeze().cpu().numpy()
    acc_learned = (preds_learned == labels_np).mean()

    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Attention weight distributions
    learned_flat = learned_attn[adj > 0].flatten()
    uniform_flat = uniform_attn[adj > 0].flatten()

    axes[0, 0].hist(learned_flat, bins=50, alpha=0.6, label='Learned', color='blue', density=True)
    axes[0, 0].hist(uniform_flat, bins=50, alpha=0.6, label='Uniform', color='orange', density=True)
    axes[0, 0].set_xlabel('Attention Weight')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Attention Weight Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Entropy comparison
    learned_entropy = compute_attention_entropy(learned_attn, adj)
    uniform_entropy = compute_attention_entropy(uniform_attn, adj)

    axes[0, 1].hist(learned_entropy, bins=30, alpha=0.6, label='Learned', color='blue', density=True)
    axes[0, 1].hist(uniform_entropy, bins=30, alpha=0.6, label='Uniform', color='orange', density=True)
    axes[0, 1].set_xlabel('Attention Entropy')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title('Attention Entropy Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Attention concentration (max attention per node)
    learned_max = np.max(learned_attn, axis=1)
    uniform_max = np.max(uniform_attn, axis=1)

    axes[1, 0].hist(learned_max, bins=30, alpha=0.6, label='Learned', color='blue', density=True)
    axes[1, 0].hist(uniform_max, bins=30, alpha=0.6, label='Uniform', color='orange', density=True)
    axes[1, 0].set_xlabel('Max Attention Weight per Node')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Attention Concentration')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Statistics
    stats_text = f"""
    Learned vs Uniform Attention Comparison:

    Learned Attention:
      Mean weight: {np.mean(learned_flat):.6f}
      Std weight:  {np.std(learned_flat):.6f}
      Mean entropy: {np.mean(learned_entropy):.4f}
      Mean max attention: {np.mean(learned_max):.4f}

    Uniform Attention:
      Mean weight: {np.mean(uniform_flat):.6f}
      Std weight:  {np.std(uniform_flat):.6f}
      Mean entropy: {np.mean(uniform_entropy):.4f}
      Mean max attention: {np.mean(uniform_max):.4f}

    Key Observations:
      - Learned attention has {'higher' if np.std(learned_flat) > np.std(uniform_flat) else 'lower'} variance
      - Learned attention is {'more' if np.mean(learned_entropy) < np.mean(uniform_entropy) else 'less'} concentrated
      - Learned max attention is {'higher' if np.mean(learned_max) > np.mean(uniform_max) else 'lower'}
    """

    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                   verticalalignment='center')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig('learned_vs_uniform_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Learned vs uniform comparison saved to learned_vs_uniform_comparison.png")


def compute_attention_entropy(attn_matrix, adj, eps=1e-8):
    """Compute entropy of attention distribution for each node"""
    n_nodes = attn_matrix.shape[0]
    entropies = []

    for i in range(n_nodes):
        # Get attention weights for neighbors
        neighbor_mask = adj[i] > 0
        if neighbor_mask.sum() > 0:
            weights = attn_matrix[i, neighbor_mask]
            # Normalize to sum to 1
            weights = weights / (weights.sum() + eps)
            # Compute entropy
            entropy = -np.sum(weights * np.log(weights + eps))
            entropies.append(entropy)
        else:
            entropies.append(0)

    return np.array(entropies)


def main():
    """Main function to run attention visualization"""
    print("="*80)
    print("GAT Attention Visualization and Analysis")
    print("="*80)

    # Load data
    dataset = 'cora'
    (features, labels, train_mask, val_mask, test_mask, biases,
     nb_nodes, ft_size, nb_classes) = load_and_prepare_data(dataset)

    # Load adjacency matrix
    adj, _, _, _, _, _, _, _ = load_data(dataset)
    adj = adj.todense()

    # Load trained model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    hid_units = [8]
    n_heads = [8, 1]

    model = GATWithAttentionCapture(
        in_features=ft_size,
        nb_classes=nb_classes,
        nb_nodes=nb_nodes,
        hid_units=hid_units,
        n_heads=n_heads,
        dropout=0.0,  # No dropout for visualization
        alpha=0.2,
        residual=False
    )

    # Load trained weights
    checkpoint_path = 'pre_trained/cora_torch/learned_seed42_best.pt'
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded model from {checkpoint_path}")
    except FileNotFoundError:
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        print("Using randomly initialized model for demonstration")

    model = model.to(device)

    # Analyze attention patterns
    print("\nAnalyzing attention patterns...")
    avg_attn, preds, labels_np, train_mask_np, adj_np = analyze_attention_patterns(
        model, features, labels, train_mask, adj, biases, device
    )

    # Generate visualizations
    print("\nGenerating visualizations...")

    # 1. Attention heatmap
    visualize_attention_heatmap(avg_attn, labels_np, sample_size=200,
                                save_path='attention_heatmap.png')

    # 2. Attention by edge type
    stats = analyze_attention_by_edge_type(avg_attn, adj_np, labels_np,
                                          save_path='attention_by_edge_type.png')

    # 3. Graph visualization
    visualize_attention_graph(avg_attn, adj_np, labels_np, features.squeeze().numpy(),
                             sample_size=80, save_path='attention_graph.png')

    # 4. Top neighbors analysis
    top_k_stats = analyze_top_attention_neighbors(avg_attn, labels_np, k=5,
                                                  save_path='top_neighbors_analysis.png')

    # 5. Compare with uniform attention
    compare_learned_vs_uniform_attention(model, features, labels, adj_np, biases, device)

    # Print summary
    print("\n" + "="*80)
    print("ATTENTION ANALYSIS SUMMARY")
    print("="*80)
    print(f"\nAttention by Edge Type:")
    print(f"  Same class edges - Mean: {stats['same_class_mean']:.6f}, Std: {stats['same_class_std']:.6f}")
    print(f"  Different class edges - Mean: {stats['diff_class_mean']:.6f}, Std: {stats['diff_class_std']:.6f}")
    print(f"  Self loops - Mean: {stats['self_loop_mean']:.6f}, Std: {stats['self_loop_std']:.6f}")

    if stats['same_class_mean'] > stats['diff_class_mean']:
        print(f"\n  ✓ Learned attention assigns higher weights to same-class neighbors!")
        print(f"    Ratio: {stats['same_class_mean'] / stats['diff_class_mean']:.2f}x")
    else:
        print(f"\n  ✗ Learned attention does not show preference for same-class neighbors")

    print(f"\nTop-5 Neighbors Analysis:")
    print(f"  Mean same-class ratio: {top_k_stats['mean_same_class_ratio']:.3f}")
    print(f"  Std same-class ratio: {top_k_stats['std_same_class_ratio']:.3f}")

    print("\n" + "="*80)
    print("Visualization files generated:")
    print("  - attention_heatmap.png")
    print("  - attention_by_edge_type.png")
    print("  - attention_graph.png")
    print("  - top_neighbors_analysis.png")
    print("  - learned_vs_uniform_comparison.png")
    print("="*80)


if __name__ == '__main__':
    main()
