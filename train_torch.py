import os
import time
import numpy as np
import torch
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils import process
import sys
sys.path.insert(0, 'models')
from gat_torch import GAT, GATAvg
from train_utils import train_step, eval_step


def train_model(model_name='gat', save_plots=True):
    dataset = 'cora'

    batch_size = 1
    nb_epochs = 100000
    patience = 100
    lr = 0.005
    l2_coef = 0.0005
    hid_units = [8]
    n_heads = [8, 1]
    residual = False
    attn_drop_rate = 0.6
    ffd_drop_rate = 0.6

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = process.load_data(dataset)
    features, spars = process.preprocess_features(features)

    nb_nodes = features.shape[0]
    ft_size = features.shape[1]
    nb_classes = y_train.shape[1]

    adj = adj.todense()
    adj = np.array(adj)

    features = features[np.newaxis]
    adj = adj[np.newaxis]
    y_train = y_train[np.newaxis]
    y_val = y_val[np.newaxis]
    y_test = y_test[np.newaxis]
    train_mask = train_mask[np.newaxis]
    val_mask = val_mask[np.newaxis]
    test_mask = test_mask[np.newaxis]

    biases = process.adj_to_bias(adj, [nb_nodes], nhood=1)

    features = torch.FloatTensor(features).to(device)
    biases = torch.FloatTensor(biases).to(device)
    adj_tensor = torch.FloatTensor(adj).to(device)
    y_train = torch.FloatTensor(y_train).to(device)
    y_val = torch.FloatTensor(y_val).to(device)
    y_test = torch.FloatTensor(y_test).to(device)
    train_mask = torch.BoolTensor(train_mask).to(device)
    val_mask = torch.BoolTensor(val_mask).to(device)
    test_mask = torch.BoolTensor(test_mask).to(device)

    adj_mask = (biases != -1e9).float().to(device)

    if model_name == 'gat':
        model = GAT(
            in_sz=ft_size,
            nb_classes=nb_classes,
            nb_nodes=nb_nodes,
            hid_units=hid_units,
            n_heads=n_heads,
            activation=torch.nn.functional.elu,
            residual=residual,
            attn_drop=attn_drop_rate,
            ffd_drop=ffd_drop_rate
        ).to(device)
        input_bias = biases
        use_avg = False
    else:
        model = GATAvg(
            in_sz=ft_size,
            nb_classes=nb_classes,
            nb_nodes=nb_nodes,
            hid_units=hid_units,
            n_heads=n_heads,
            activation=torch.nn.functional.elu,
            residual=residual,
            in_drop=ffd_drop_rate
        ).to(device)
        input_bias = adj_mask
        use_avg = True

    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"Model: {model_name}")
    print('Dataset: ' + dataset)
    print('----- Opt. hyperparams -----')
    print('lr: ' + str(lr))
    print('l2_coef: ' + str(l2_coef))
    print('----- Archi. hyperparams -----')
    print('nb. layers: ' + str(len(hid_units)))
    print('nb. units per layer: ' + str(hid_units))
    print('nb. attention heads: ' + str(n_heads))
    print('residual: ' + str(residual))

    vlss_mn = float('inf')
    vacc_mx = 0.0
    curr_step = 0

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    best_val_loss = float('inf')
    best_val_acc = 0.0

    for epoch in range(nb_epochs):
        train_loss_avg = 0
        train_acc_avg = 0
        val_loss_avg = 0
        val_acc_avg = 0
        tr_step = 0

        loss_tr, acc_tr = train_step(
            model, optimizer, features, input_bias, y_train, train_mask,
            l2_coef, attn_drop_rate, ffd_drop_rate, device, use_avg
        )
        train_loss_avg += loss_tr
        train_acc_avg += acc_tr
        tr_step += 1

        vl_step = 0
        loss_vl, acc_vl = eval_step(model, features, input_bias, y_val, val_mask, device, use_avg)
        val_loss_avg += loss_vl
        val_acc_avg += acc_vl
        vl_step += 1

        print(f'Training: loss = %.5f, acc = %.5f | Val: loss = %.5f, acc = %.5f' %
                (train_loss_avg/tr_step, train_acc_avg/tr_step,
                val_loss_avg/vl_step, val_acc_avg/vl_step))

        train_losses.append(train_loss_avg/tr_step)
        train_accs.append(train_acc_avg/tr_step)
        val_losses.append(val_loss_avg/vl_step)
        val_accs.append(val_acc_avg/vl_step)

        if val_acc_avg/vl_step >= vacc_mx or val_loss_avg/vl_step <= vlss_mn:
            if val_acc_avg/vl_step >= vacc_mx and val_loss_avg/vl_step <= vlss_mn:
                vacc_early_model = val_acc_avg/vl_step
                vlss_early_model = val_loss_avg/vl_step
                checkpt_file = f'pre_trained/cora/{model_name}_best.pth'
                torch.save(model.state_dict(), checkpt_file)

                best_val_loss = val_loss_avg/vl_step
                best_val_acc = val_acc_avg/vl_step

            vacc_mx = np.max([val_acc_avg/vl_step, vacc_mx])
            vlss_mn = np.min([val_loss_avg/vl_step, vlss_mn])
            curr_step = 0
        else:
            curr_step += 1
            if curr_step == patience:
                print('Early stop! Min loss: ', vlss_mn, ', Max accuracy: ', vacc_mx)
                print('Early stop model validation loss: ', vlss_early_model, ', accuracy: ', vacc_early_model)
                break

    checkpt_file = f'pre_trained/cora/{model_name}_best.pth'
    model.load_state_dict(torch.load(checkpt_file))

    ts_loss, ts_acc = eval_step(model, features, input_bias, y_test, test_mask, device, use_avg)

    print(f'{model_name} Test loss:', ts_loss, '; Test accuracy:', ts_acc)

    if save_plots:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].plot(train_losses, label='Train Loss')
        axes[0].plot(val_losses, label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title(f'{model_name.upper()} Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(train_accs, label='Train Acc')
        axes[1].plot(val_accs, label='Val Acc')
        axes[1].axhline(y=ts_acc, color='r', linestyle='--', label=f'Test Acc={ts_acc:.4f}')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title(f'{model_name.upper()} Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.savefig(f'{model_name}_training_curves.png')
        plt.close()

    return train_losses, train_accs, val_losses, val_accs, ts_loss, ts_acc


if __name__ == '__main__':
    print("="*50)
    print("Training GAT with attention (baseline)...")
    print("="*50)
    gat_results = train_model(model_name='gat', save_plots=True)

    print("\n" + "="*50)
    print("Training GAT with average (no attention)...")
    print("="*50)
    avg_results = train_model(model_name='avg', save_plots=True)

    gat_train_loss, gat_train_acc, gat_val_loss, gat_val_acc, gat_test_loss, gat_test_acc = gat_results
    avg_train_loss, avg_train_acc, avg_val_loss, avg_val_acc, avg_test_loss, avg_test_acc = avg_results

    print("\n" + "="*50)
    print("Experiment Summary")
    print("="*50)
    print(f"GAT (with attention):")
    print(f"  Test Accuracy: {gat_test_acc:.4f}")
    print(f"  Test Loss: {gat_test_loss:.4f}")
    print(f"\nGAT-Avg (without attention, uniform weights):")
    print(f"  Test Accuracy: {avg_test_acc:.4f}")
    print(f"  Test Loss: {avg_test_loss:.4f}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(gat_train_loss, label='Train Loss')
    axes[0, 0].plot(gat_val_loss, label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('GAT (with attention) Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    axes[0, 1].plot(avg_train_loss, label='Train Loss', color='orange')
    axes[0, 1].plot(avg_val_loss, label='Val Loss', color='red')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('GAT-Avg (uniform weights) Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    axes[1, 0].plot(gat_train_acc, label='Train Acc')
    axes[1, 0].plot(gat_val_acc, label='Val Acc')
    axes[1, 0].axhline(y=gat_test_acc, color='r', linestyle='--', label=f'Test Acc={gat_test_acc:.4f}')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_title('GAT (with attention) Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    axes[1, 1].plot(avg_train_acc, label='Train Acc', color='orange')
    axes[1, 1].plot(avg_val_acc, label='Val Acc', color='red')
    axes[1, 1].axhline(y=avg_test_acc, color='purple', linestyle='--', label=f'Test Acc={avg_test_acc:.4f}')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_title('GAT-Avg (uniform weights) Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig('comparison_experiment.png')
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(gat_train_loss, label='GAT Train')
    axes[0].plot(avg_train_loss, label='GAT-Avg Train', linestyle='--')
    axes[0].plot(gat_val_loss, label='GAT Val')
    axes[0].plot(avg_val_loss, label='GAT-Avg Val', linestyle='--')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Comparison')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(gat_train_acc, label='GAT Train')
    axes[1].plot(avg_train_acc, label='GAT-Avg Train', linestyle='--')
    axes[1].plot(gat_val_acc, label='GAT Val')
    axes[1].plot(avg_val_acc, label='GAT-Avg Val', linestyle='--')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy Comparison')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('direct_comparison.png')
    plt.close()

    print("\nPlots saved as:")
    print("  - gat_training_curves.png")
    print("  - avg_training_curves.png")
    print("  - comparison_experiment.png")
    print("  - direct_comparison.png")
