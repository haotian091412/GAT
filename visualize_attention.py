import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import importlib.util

def load_module_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

process = load_module_from_path('process', 'utils/process.py')
gat_torch = load_module_from_path('gat_torch', 'models/gat_torch.py')
GATTorch = gat_torch.GATTorch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = process.load_data('cora')
features, spars = process.preprocess_features(features)

nb_nodes = features.shape[0]
ft_size = features.shape[1]
nb_classes = y_train.shape[1]

adj_dense = adj.todense()

features = features[np.newaxis]
adj_dense = adj_dense[np.newaxis]

biases = process.adj_to_bias(adj_dense, [nb_nodes], nhood=1)

features_tensor = torch.FloatTensor(features).to(device)
biases_tensor = torch.FloatTensor(biases).to(device)
adj_tensor = torch.FloatTensor(adj_dense).to(device)

def get_attention_weights(model, features, bias_mat):
    model.eval()
    attn_weights = []
    
    def hook_fn(name):
        def hook(module, input, output):
            seq = input[0]
            seq_fts = module.conv_seq(seq.transpose(1, 2)).transpose(1, 2)
            f_1 = module.conv_f1(seq_fts.transpose(1, 2)).transpose(1, 2)
            f_2 = module.conv_f2(seq_fts.transpose(1, 2)).transpose(1, 2)
            logits = f_1 + f_2.transpose(1, 2)
            coefs = torch.softmax(torch.nn.functional.leaky_relu(logits) + bias_mat, dim=-1)
            attn_weights.append((name, coefs.detach().cpu().numpy()))
        return hook
    
    hooks = []
    for i, layer_heads in enumerate(model.layers):
        for j, head in enumerate(layer_heads):
            h = head.register_forward_hook(hook_fn(f"layer_{i}_head_{j}"))
            hooks.append(h)
    
    with torch.no_grad():
        _ = model(features, bias_mat)
    
    for h in hooks:
        h.remove()
    
    return attn_weights

print("="*60)
print("Loading pre-trained GAT model...")
print("="*60)

hid_units = [8]
n_heads = [8, 1]

model = GATTorch(
    in_sz=ft_size,
    nb_classes=nb_classes,
    nb_nodes=nb_nodes,
    hid_units=hid_units,
    n_heads=n_heads,
    activation=torch.nn.functional.elu,
    residual=False,
    attn_drop=0.6,
    ffd_drop=0.6
).to(device)

try:
    model.load_state_dict(torch.load('pre_trained/cora/gat_best.pth', map_location=device))
    print("Loaded pre-trained weights successfully")
except:
    print("Using random weights for demonstration")

print("\nExtracting attention weights...")
attn_weights = get_attention_weights(model, features_tensor, biases_tensor)

print("\nGenerating attention visualization...")

num_heads_to_show = min(4, len(attn_weights))
sample_nodes = 5
node_ids = np.random.choice(nb_nodes, sample_nodes, replace=False)

plt.style.use('seaborn-v0_8')

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

for idx, (name, weights) in enumerate(attn_weights[:num_heads_to_show]):
    weights_sample = weights[0]
    
    for node_id in node_ids[:1]:
        neighbor_weights = weights_sample[node_id]
        top_k = 10
        top_indices = np.argsort(neighbor_weights)[::-1][:top_k]
        top_weights = neighbor_weights[top_indices]
        
        axes[idx].bar(range(top_k), top_weights, alpha=0.7, edgecolor='black')
        axes[idx].set_xlabel('Top Neighbors (by attention weight)')
        axes[idx].set_ylabel('Attention Weight')
        axes[idx].set_title(f'{name}: Attention Distribution for Node {node_id}')
        axes[idx].set_ylim([0, max(top_weights) * 1.1])
        
        for i, (idx_n, w) in enumerate(zip(range(top_k), top_weights)):
            axes[idx].text(idx_n, w + max(top_weights)*0.02, f'{w:.3f}', 
                          ha='center', fontsize=8, rotation=0)

plt.suptitle('Attention Weights Distribution (First Layer - Different Heads)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('attention_distribution_heads.png', dpi=300, bbox_inches='tight')
plt.close()
print("  - attention_distribution_heads.png (attention weights per head)")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

name, weights = attn_weights[0]
weights_sample = weights[0]

node_degrees = np.array(adj.sum(axis=1)).flatten()
high_deg_node = np.argmax(node_degrees)
low_deg_node = np.argmin(node_degrees + 1e6 * (node_degrees == 0))
rand_node = np.random.choice(nb_nodes)

for ax, node_id, title in zip(axes, [high_deg_node, low_deg_node, rand_node], 
                             ['High-Degree Node', 'Low-Degree Node', 'Random Node']):
    neighbor_weights = weights_sample[node_id]
    deg = int(node_degrees[node_id])
    top_k = min(deg + 1, 15)
    top_indices = np.argsort(neighbor_weights)[::-1][:top_k]
    top_weights = neighbor_weights[top_indices]
    
    bars = ax.bar(range(top_k), top_weights, alpha=0.7, edgecolor='black', color='#1f77b4')
    ax.set_xlabel('Neighbors (sorted by attention weight)')
    ax.set_ylabel('Attention Weight')
    ax.set_title(f'{title} (degree={deg})')
    ax.set_ylim([0, max(top_weights) * 1.2 if max(top_weights) > 0 else 0.1])

plt.suptitle('Attention Distribution for Different Node Types', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('attention_by_node_type.png', dpi=300, bbox_inches='tight')
plt.close()
print("  - attention_by_node_type.png (attention for different node types)")

fig, ax = plt.subplots(figsize=(10, 8))

head_stds = []
head_means = []
head_names = []

for name, weights in attn_weights:
    weights_flat = weights[0].flatten()
    head_stds.append(np.std(weights_flat))
    head_means.append(np.mean(weights_flat))
    head_names.append(name)

x_pos = np.arange(len(head_stds[:8]))
ax.bar(x_pos, head_stds[:8], alpha=0.7, edgecolor='black', label='Std Dev')
ax.set_xlabel('Attention Head')
ax.set_ylabel('Standard Deviation of Attention Weights')
ax.set_title('Variability of Attention Weights Across Different Heads')
ax.set_xticks(x_pos)
ax.set_xticklabels([f'Head {i}' for i in range(8)])
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('attention_head_variability.png', dpi=300, bbox_inches='tight')
plt.close()
print("  - attention_head_variability.png (head variability)")

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

name, weights = attn_weights[0]
weights_2d = weights[0][:50, :50]

mask = adj_dense[0][:50, :50]

for head_idx, ax in enumerate(axes):
    if head_idx >= len(attn_weights):
        break
    name, weights = attn_weights[head_idx]
    weights_2d = weights[0][:50, :50]
    weights_masked = weights_2d * mask
    
    im = ax.imshow(weights_masked, cmap='hot', interpolation='nearest')
    ax.set_title(f'Head {head_idx}')
    ax.set_xticks([])
    ax.set_yticks([])

plt.suptitle('Attention Matrix Heatmap (First 50 Nodes x 50 Nodes)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('attention_matrix_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("  - attention_matrix_heatmap.png (attention matrix)")

print("\n" + "="*60)
print("Statistical Analysis")
print("="*60)

with open('attention_analysis.txt', 'w') as f:
    f.write("="*60 + "\n")
    f.write("Attention Weight Analysis\n")
    f.write("="*60 + "\n\n")
    
    for name, weights in attn_weights:
        w = weights[0]
        non_zero = w[w > 1e-6]
        f.write(f"\n{name}:\n")
        f.write(f"  Mean attention weight: {np.mean(non_zero):.6f}\n")
        f.write(f"  Std attention weight:  {np.std(non_zero):.6f}\n")
        f.write(f"  Max attention weight:   {np.max(w):.6f}\n")
        f.write(f"  Min attention weight:   {np.min(non_zero):.6f}\n")
        f.write(f"  Sparsity (near-zero):   {np.sum(w < 1e-6) / w.size * 100:.2f}%\n")
        
        print(f"\n{name}:")
        print(f"  Mean: {np.mean(non_zero):.6f}, Std: {np.std(non_zero):.6f}")
        print(f"  Max: {np.max(w):.6f}, Min: {np.min(non_zero):.6f}")

print("\nVisualization complete! Generated files:")
print("  - attention_distribution_heads.png")
print("  - attention_by_node_type.png")
print("  - attention_head_variability.png")
print("  - attention_matrix_heatmap.png")
print("  - attention_analysis.txt")
