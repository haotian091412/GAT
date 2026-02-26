import time
import numpy as np
import torch
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
import importlib.util

def load_module_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

process = load_module_from_path('process', 'utils/process.py')
base_gattn_torch = load_module_from_path('base_gattn_torch', 'models/base_gattn_torch.py')
masked_softmax_cross_entropy = base_gattn_torch.masked_softmax_cross_entropy
masked_accuracy = base_gattn_torch.masked_accuracy

gat_torch = load_module_from_path('gat_torch', 'models/gat_torch.py')
GATTorch = gat_torch.GATTorch

gat_avg_torch = load_module_from_path('gat_avg_torch', 'models/gat_avg_torch.py')
GATAvgTorch = gat_avg_torch.GATAvgTorch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

dataset = 'cora'

batch_size = 1
nb_epochs = 100000
patience = 100
lr = 0.005
l2_coef = 0.0005
hid_units = [8]
n_heads = [8, 1]
residual = False

print('Dataset: ' + dataset)
print('----- Opt. hyperparams -----')
print('lr: ' + str(lr))
print('l2_coef: ' + str(l2_coef))
print('----- Archi. hyperparams -----')
print('nb. layers: ' + str(len(hid_units)))
print('nb. units per layer: ' + str(hid_units))
print('nb. attention heads: ' + str(n_heads))
print('residual: ' + str(residual))

adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = process.load_data(dataset)
features, spars = process.preprocess_features(features)

nb_nodes = features.shape[0]
ft_size = features.shape[1]
nb_classes = y_train.shape[1]

adj_dense = adj.todense()

features = features[np.newaxis]
adj_dense = adj_dense[np.newaxis]
y_train = y_train[np.newaxis]
y_val = y_val[np.newaxis]
y_test = y_test[np.newaxis]
train_mask = train_mask[np.newaxis]
val_mask = val_mask[np.newaxis]
test_mask = test_mask[np.newaxis]

biases = process.adj_to_bias(adj_dense, [nb_nodes], nhood=1)
adj_for_const = adj_dense.copy()

features_tensor = torch.FloatTensor(features).to(device)
biases_tensor = torch.FloatTensor(biases).to(device)
adj_tensor = torch.FloatTensor(adj_for_const).to(device)
y_train_tensor = torch.FloatTensor(y_train).to(device)
y_val_tensor = torch.FloatTensor(y_val).to(device)
y_test_tensor = torch.FloatTensor(y_test).to(device)
train_mask_tensor = torch.BoolTensor(train_mask).to(device)
val_mask_tensor = torch.BoolTensor(val_mask).to(device)
test_mask_tensor = torch.BoolTensor(test_mask).to(device)


def train_model(model, model_name, use_bias=True, checkpt_file=None):
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)
    
    vlss_mn = np.inf
    vacc_mx = 0.0
    curr_step = 0
    
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []
    
    for epoch in range(nb_epochs):
        model.train()
        optimizer.zero_grad()
        
        if use_bias:
            logits = model(features_tensor, biases_tensor)
        else:
            logits = model(features_tensor, adj_tensor)
        
        log_resh = logits.view(-1, nb_classes)
        lab_resh = y_train_tensor.view(-1, nb_classes)
        msk_resh = train_mask_tensor.view(-1)
        
        loss = masked_softmax_cross_entropy(log_resh, lab_resh, msk_resh)
        acc = masked_accuracy(log_resh, lab_resh, msk_resh)
        
        loss.backward()
        optimizer.step()
        
        train_loss_val = loss.item()
        train_acc_val = acc.item()
        
        model.eval()
        with torch.no_grad():
            if use_bias:
                logits_val = model(features_tensor, biases_tensor)
            else:
                logits_val = model(features_tensor, adj_tensor)
            
            log_resh_val = logits_val.view(-1, nb_classes)
            lab_resh_val = y_val_tensor.view(-1, nb_classes)
            msk_resh_val = val_mask_tensor.view(-1)
            
            val_loss = masked_softmax_cross_entropy(log_resh_val, lab_resh_val, msk_resh_val)
            val_acc = masked_accuracy(log_resh_val, lab_resh_val, msk_resh_val)
            
            val_loss_val = val_loss.item()
            val_acc_val = val_acc.item()
        
        train_loss_history.append(train_loss_val)
        train_acc_history.append(train_acc_val)
        val_loss_history.append(val_loss_val)
        val_acc_history.append(val_acc_val)
        
        if epoch % 20 == 0:
            print(f'Epoch {epoch}: Training: loss = {train_loss_val:.5f}, acc = {train_acc_val:.5f} | Val: loss = {val_loss_val:.5f}, acc = {val_acc_val:.5f}')
        
        if val_acc_val >= vacc_mx or val_loss_val <= vlss_mn:
            if val_acc_val >= vacc_mx and val_loss_val <= vlss_mn:
                vacc_early_model = val_acc_val
                vlss_early_model = val_loss_val
                if checkpt_file:
                    torch.save(model.state_dict(), checkpt_file)
            vacc_mx = np.max((val_acc_val, vacc_mx))
            vlss_mn = np.min((val_loss_val, vlss_mn))
            curr_step = 0
        else:
            curr_step += 1
            if curr_step == patience:
                print(f'Early stop! Min loss: {vlss_mn}, Max accuracy: {vacc_mx}')
                print(f'Early stop model validation loss: {vlss_early_model}, accuracy: {vacc_early_model}')
                break
    
    if checkpt_file:
        model.load_state_dict(torch.load(checkpt_file))
    
    model.eval()
    with torch.no_grad():
        if use_bias:
            logits_test = model(features_tensor, biases_tensor)
        else:
            logits_test = model(features_tensor, adj_tensor)
        
        log_resh_test = logits_test.view(-1, nb_classes)
        lab_resh_test = y_test_tensor.view(-1, nb_classes)
        msk_resh_test = test_mask_tensor.view(-1)
        
        ts_loss = masked_softmax_cross_entropy(log_resh_test, lab_resh_test, msk_resh_test)
        ts_acc = masked_accuracy(log_resh_test, lab_resh_test, msk_resh_test)
    
    print(f'Test loss: {ts_loss.item():.5f}; Test accuracy: {ts_acc.item():.5f}')
    
    return {
        'train_loss': train_loss_history,
        'train_acc': train_acc_history,
        'val_loss': val_loss_history,
        'val_acc': val_acc_history,
        'test_loss': ts_loss.item(),
        'test_acc': ts_acc.item()
    }


results = {}

model_avg = GATAvgTorch(
    in_sz=ft_size,
    nb_classes=nb_classes,
    nb_nodes=nb_nodes,
    hid_units=hid_units,
    n_heads=n_heads,
    activation=torch.nn.functional.elu,
    residual=residual,
    attn_drop=0.6,
    ffd_drop=0.6
)
results['GAT-Avg (邻域平均)'] = train_model(model_avg, 'GAT-Avg (邻域平均)', use_bias=False, checkpt_file='pre_trained/cora/gat_avg_best.pth')

model_single = GATTorch(
    in_sz=ft_size,
    nb_classes=nb_classes,
    nb_nodes=nb_nodes,
    hid_units=hid_units,
    n_heads=[1, 1],
    activation=torch.nn.functional.elu,
    residual=residual,
    attn_drop=0.6,
    ffd_drop=0.6
)
results['GAT-Single (单头注意力)'] = train_model(model_single, 'GAT-Single (单头注意力)', use_bias=True, checkpt_file='pre_trained/cora/gat_single_best.pth')

model_gat = GATTorch(
    in_sz=ft_size,
    nb_classes=nb_classes,
    nb_nodes=nb_nodes,
    hid_units=hid_units,
    n_heads=n_heads,
    activation=torch.nn.functional.elu,
    residual=residual,
    attn_drop=0.6,
    ffd_drop=0.6
)
results['GAT (8头注意力)'] = train_model(model_gat, 'GAT (8头注意力)', use_bias=True, checkpt_file='pre_trained/cora/gat_best.pth')

print("\n\n" + "="*80)
print("实验结果总结")
print("="*80)

with open('experiment_results_torch.txt', 'w') as f:
    for model_name, res in results.items():
        print(f"\n{model_name}:")
        print(f"  测试集损失: {res['test_loss']:.5f}")
        print(f"  测试集准确率: {res['test_acc']:.5f}")
        f.write(f"{model_name}:\n")
        f.write(f"  测试集损失: {res['test_loss']:.5f}\n")
        f.write(f"  测试集准确率: {res['test_acc']:.5f}\n\n")

plt.style.use('seaborn-v0_8')
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
labels = list(results.keys())

for idx, (model_name, res) in enumerate(results.items()):
    axes[0, 0].plot(res['train_loss'], label=model_name, color=colors[idx], alpha=0.8)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Training Loss')
axes[0, 0].set_title('Training Loss Comparison')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

for idx, (model_name, res) in enumerate(results.items()):
    axes[0, 1].plot(res['train_acc'], label=model_name, color=colors[idx], alpha=0.8)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Training Accuracy')
axes[0, 1].set_title('Training Accuracy Comparison')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

for idx, (model_name, res) in enumerate(results.items()):
    axes[1, 0].plot(res['val_loss'], label=model_name, color=colors[idx], alpha=0.8)
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Validation Loss')
axes[1, 0].set_title('Validation Loss Comparison')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

for idx, (model_name, res) in enumerate(results.items()):
    axes[1, 1].plot(res['val_acc'], label=model_name, color=colors[idx], alpha=0.8)
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Validation Accuracy')
axes[1, 1].set_title('Validation Accuracy Comparison')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_curves_torch.png', dpi=300, bbox_inches='tight')
plt.close()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

test_accs = [res['test_acc'] for res in results.values()]
x_pos = np.arange(len(labels))
bars = ax1.bar(x_pos, test_accs, color=colors, alpha=0.7, edgecolor='black')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(labels, rotation=15, ha='right', fontsize=9)
ax1.set_ylabel('测试集准确率')
ax1.set_title('Test Accuracy Comparison')
ax1.set_ylim([min(test_accs) - 0.05, max(test_accs) + 0.02])
ax1.grid(True, alpha=0.3, axis='y')
for i, v in enumerate(test_accs):
    ax1.text(i, v + 0.003, f'{v:.4f}', ha='center', fontweight='bold')

test_losses = [res['test_loss'] for res in results.values()]
bars2 = ax2.bar(x_pos, test_losses, color=colors, alpha=0.7, edgecolor='black')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(labels, rotation=15, ha='right', fontsize=9)
ax2.set_ylabel('测试集损失')
ax2.set_title('Test Loss Comparison')
ax2.grid(True, alpha=0.3, axis='y')
for i, v in enumerate(test_losses):
    ax2.text(i, v + 0.01, f'{v:.4f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('test_comparison_torch.png', dpi=300, bbox_inches='tight')
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for idx, (model_name, res) in enumerate(results.items()):
    axes[0].plot(res['train_acc'], label=model_name, color=colors[idx], linewidth=2)
axes[0].set_xlabel('训练轮次 (Epoch)')
axes[0].set_ylabel('准确率')
axes[0].set_title('Training Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

for idx, (model_name, res) in enumerate(results.items()):
    axes[1].plot(res['val_acc'], label=model_name, color=colors[idx], linewidth=2)
axes[1].set_xlabel('训练轮次 (Epoch)')
axes[1].set_ylabel('准确率')
axes[1].set_title('Validation Accuracy')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('accuracy_comparison_torch.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n图表已保存为:")
print("  - training_curves_torch.png (训练/验证损失和准确率曲线)")
print("  - test_comparison_torch.png (测试集结果柱状图)")
print("  - accuracy_comparison_torch.png (准确率对比)")
print("  - experiment_results_torch.txt (实验数值结果)")
