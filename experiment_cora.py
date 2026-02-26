import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, '/Users/haotianma/Documents/gsb_project/0225/glm/GAT')
from utils.process import load_data, preprocess_features, adj_to_bias

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models'))
from gat_pytorch import GAT, NoAttnGAT, SingleHeadGAT

dataset = 'cora'

batch_size = 1
nb_epochs = 100000
patience = 100
lr = 0.005
l2_coef = 0.0005
hid_units = [8]
n_heads = [8, 1]
residual = False
dropout = 0.6
alpha = 0.2

print('Dataset: ' + dataset)
print('----- Opt. hyperparams -----')
print('lr: ' + str(lr))
print('l2_coef: ' + str(l2_coef))
print('----- Archi. hyperparams -----')
print('nb. layers: ' + str(len(hid_units)))
print('nb. units per layer: ' + str(hid_units))
print('nb. attention heads: ' + str(n_heads))
print('residual: ' + str(residual))
print('dropout: ' + str(dropout))

adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(dataset)
features, spars = preprocess_features(features)

nb_nodes = features.shape[0]
ft_size = features.shape[1]
nb_classes = y_train.shape[1]

adj = adj.todense()

features = features[np.newaxis]
adj = adj[np.newaxis]
y_train = y_train[np.newaxis]
y_val = y_val[np.newaxis]
y_test = y_test[np.newaxis]
train_mask = train_mask[np.newaxis]
val_mask = val_mask[np.newaxis]
test_mask = test_mask[np.newaxis]

biases = adj_to_bias(adj, [nb_nodes], nhood=1)

features = torch.FloatTensor(features)
biases = torch.FloatTensor(biases)
y_train = torch.FloatTensor(y_train)
y_val = torch.FloatTensor(y_val)
y_test = torch.FloatTensor(y_test)
train_mask = torch.BoolTensor(train_mask)
val_mask = torch.BoolTensor(val_mask)
test_mask = torch.BoolTensor(test_mask)

def masked_softmax_cross_entropy(logits, labels, mask):
    logits_reshaped = logits.view(-1, logits.size(-1))
    labels_reshaped = labels.view(-1, labels.size(-1))
    mask_reshaped = mask.view(-1)
    
    loss = nn.CrossEntropyLoss(reduction='none')(logits_reshaped, labels_reshaped.argmax(dim=1))
    mask_reshaped = mask_reshaped.float()
    mask_reshaped = mask_reshaped / mask_reshaped.mean()
    loss = loss * mask_reshaped
    return loss.mean()

def masked_accuracy(logits, labels, mask):
    preds = logits.argmax(dim=2)
    correct = (preds == labels.argmax(dim=2)).float()
    mask = mask.float()
    mask = mask / mask.mean()
    correct = correct * mask
    return correct.mean()

def train_model(model, model_name, features, biases, y_train, y_val, y_test, 
                train_mask, val_mask, test_mask, lr, l2_coef, nb_epochs, patience):
    print(f'\n===== Training {model_name} =====')
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    vlss_mn = np.inf
    vacc_mx = 0.0
    curr_step = 0
    best_model_state = None
    
    for epoch in range(nb_epochs):
        model.train()
        optimizer.zero_grad()
        
        logits = model(features, biases)
        
        train_loss = masked_softmax_cross_entropy(logits, y_train, train_mask)
        train_acc = masked_accuracy(logits, y_train, train_mask)
        
        train_loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            logits_val = model(features, biases)
            val_loss = masked_softmax_cross_entropy(logits_val, y_val, val_mask)
            val_acc = masked_accuracy(logits_val, y_val, val_mask)
        
        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())
        train_accs.append(train_acc.item())
        val_accs.append(val_acc.item())
        
        if (epoch + 1) % 100 == 0:
            print(f'Epoch {epoch+1}: Train Loss = {train_loss.item():.5f}, Train Acc = {train_acc.item():.5f} | '
                  f'Val Loss = {val_loss.item():.5f}, Val Acc = {val_acc.item():.5f}')
        
        if val_acc.item() >= vacc_mx or val_loss.item() <= vlss_mn:
            if val_acc.item() >= vacc_mx and val_loss.item() <= vlss_mn:
                best_model_state = model.state_dict().copy()
            vacc_mx = max(val_acc.item(), vacc_mx)
            vlss_mn = min(val_loss.item(), vlss_mn)
            curr_step = 0
        else:
            curr_step += 1
            if curr_step == patience:
                print(f'Early stop at epoch {epoch+1}! Min loss: {vlss_mn:.5f}, Max accuracy: {vacc_mx:.5f}')
                break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    model.eval()
    with torch.no_grad():
        logits_test = model(features, biases)
        test_loss = masked_softmax_cross_entropy(logits_test, y_test, test_mask)
        test_acc = masked_accuracy(logits_test, y_test, test_mask)
    
    print(f'Test Loss: {test_loss.item():.5f}, Test Accuracy: {test_acc.item():.5f}')
    
    return train_losses, val_losses, train_accs, val_accs, test_loss.item(), test_acc.item()

print('\n' + '='*50)
print('Experiment 1: Standard GAT (Multi-Head Attention)')
print('='*50)
model_gat = GAT(nfeat=ft_size, nhid=hid_units, nclass=nb_classes, n_heads=n_heads, dropout=dropout, alpha=alpha)
gat_results = train_model(model_gat, 'Standard GAT', features, biases, y_train, y_val, y_test,
                          train_mask, val_mask, test_mask, lr, l2_coef, nb_epochs, patience)

print('\n' + '='*50)
print('Experiment 2: No-Attention GAT (Fixed Uniform Weights)')
print('='*50)
model_noattn = NoAttnGAT(nfeat=ft_size, nhid=hid_units, nclass=nb_classes, n_heads=n_heads, dropout=dropout)
noattn_results = train_model(model_noattn, 'No-Attention GAT', features, biases, y_train, y_val, y_test,
                              train_mask, val_mask, test_mask, lr, l2_coef, nb_epochs, patience)

print('\n' + '='*50)
print('Experiment 3: Single-Head GAT')
print('='*50)
model_single = SingleHeadGAT(nfeat=ft_size, nhid=hid_units, nclass=nb_classes, dropout=dropout, alpha=alpha)
single_results = train_model(model_single, 'Single-Head GAT', features, biases, y_train, y_val, y_test,
                              train_mask, val_mask, test_mask, lr, l2_coef, nb_epochs, patience)

plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.plot(gat_results[0], label='Standard GAT', color='blue')
plt.plot(noattn_results[0], label='No-Attention GAT', color='red')
plt.plot(single_results[0], label='Single-Head GAT', color='green')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Training Loss Curves')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(gat_results[1], label='Standard GAT', color='blue')
plt.plot(noattn_results[1], label='No-Attention GAT', color='red')
plt.plot(single_results[1], label='Single-Head GAT', color='green')
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.title('Validation Loss Curves')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(gat_results[2], label='Standard GAT', color='blue')
plt.plot(noattn_results[2], label='No-Attention GAT', color='red')
plt.plot(single_results[2], label='Single-Head GAT', color='green')
plt.xlabel('Epoch')
plt.ylabel('Training Accuracy')
plt.title('Training Accuracy Curves')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(gat_results[3], label='Standard GAT', color='blue')
plt.plot(noattn_results[3], label='No-Attention GAT', color='red')
plt.plot(single_results[3], label='Single-Head GAT', color='green')
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.title('Validation Accuracy Curves')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_curves_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

print('\n' + '='*50)
print('EXPERIMENT SUMMARY')
print('='*50)
print(f'\n{"Model":<25} {"Test Loss":<15} {"Test Accuracy":<15}')
print('-'*55)
print(f'{"Standard GAT":<25} {gat_results[4]:<15.5f} {gat_results[5]:<15.5f}')
print(f'{"No-Attention GAT":<25} {noattn_results[4]:<15.5f} {noattn_results[5]:<15.5f}')
print(f'{"Single-Head GAT":<25} {single_results[4]:<15.5f} {single_results[5]:<15.5f}')
print('-'*55)

plt.figure(figsize=(10, 6))
models = ['Standard GAT\n(Multi-Head)', 'No-Attention GAT\n(Fixed Weights)', 'Single-Head GAT']
test_accs = [gat_results[5], noattn_results[5], single_results[5]]
colors = ['blue', 'red', 'green']

bars = plt.bar(models, test_accs, color=colors, alpha=0.7, edgecolor='black')
plt.ylabel('Test Accuracy')
plt.title('Test Accuracy Comparison: Attention Mechanism Impact')
plt.ylim(0, 1)

for bar, acc in zip(bars, test_accs):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{acc:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('test_accuracy_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

print('\nTraining curves saved to: training_curves_comparison.png')
print('Test accuracy comparison saved to: test_accuracy_comparison.png')
