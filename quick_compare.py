"""
Quick comparison of Learned vs Uniform Attention with fewer epochs
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from gat_pytorch import GAT, masked_softmax_cross_entropy, masked_accuracy, load_and_prepare_data
from experiments import GATVariant


def train_and_evaluate(attention_type, seed=42, nb_epochs=10000, patience=100):
    """Train and evaluate a model"""
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"\n{'='*60}")
    print(f"Training {attention_type.upper()} ATTENTION (seed={seed})")
    print(f"{'='*60}")

    # Hyperparameters
    dataset = 'cora'
    hid_units = [8]
    n_heads = [8, 1]
    lr = 0.005
    l2_coef = 0.0005
    dropout = 0.6
    alpha = 0.2

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
    model = model.to(device)
    features = features.to(device)
    labels = labels.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)
    biases = biases.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)

    best_val_acc = 0.0
    best_val_loss = float('inf')
    curr_step = 0
    best_model_state = None

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(nb_epochs):
        model.train()
        optimizer.zero_grad()

        logits = model(features, biases)
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
            val_logits = model(features, biases)
            val_logits_reshaped = val_logits.reshape(-1, val_logits.shape[-1])
            val_mask_reshaped = val_mask.reshape(-1)

            val_loss = masked_softmax_cross_entropy(val_logits_reshaped, labels_reshaped, val_mask_reshaped)
            val_acc = masked_accuracy(val_logits_reshaped, labels_reshaped, val_mask_reshaped)

        history['train_loss'].append(loss.item())
        history['train_acc'].append(acc.item())
        history['val_loss'].append(val_loss.item())
        history['val_acc'].append(val_acc.item())

        if epoch % 100 == 0:
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
                print(f'Early stop at epoch {epoch}!')
                break

    # Load best model and test
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    model.eval()
    with torch.no_grad():
        test_logits = model(features, biases)
        test_logits_reshaped = test_logits.reshape(-1, test_logits.shape[-1])
        test_mask_reshaped = test_mask.reshape(-1)

        test_loss = masked_softmax_cross_entropy(test_logits_reshaped, labels_reshaped, test_mask_reshaped)
        test_acc = masked_accuracy(test_logits_reshaped, labels_reshaped, test_mask_reshaped)

    print(f'Final Test accuracy: {test_acc.item()*100:.2f}%')

    return history, test_acc.item()


def main():
    """Main function"""
    print("="*80)
    print("QUICK COMPARISON: LEARNED vs UNIFORM ATTENTION")
    print("="*80)

    seeds = [42, 123, 456]
    learned_accs = []
    uniform_accs = []
    learned_histories = []
    uniform_histories = []

    for seed in seeds:
        # Learned attention
        history_learned, acc_learned = train_and_evaluate('learned', seed=seed, nb_epochs=5000, patience=100)
        learned_accs.append(acc_learned)
        learned_histories.append(history_learned)

        # Uniform attention
        history_uniform, acc_uniform = train_and_evaluate('uniform', seed=seed, nb_epochs=5000, patience=100)
        uniform_accs.append(acc_uniform)
        uniform_histories.append(history_uniform)

    # Statistics
    learned_mean = np.mean(learned_accs)
    learned_std = np.std(learned_accs)
    uniform_mean = np.mean(uniform_accs)
    uniform_std = np.std(uniform_accs)

    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"\nLearned Attention:")
    print(f"  Accuracies: {[f'{acc*100:.2f}%' for acc in learned_accs]}")
    print(f"  Mean: {learned_mean*100:.2f}% ± {learned_std*100:.2f}%")

    print(f"\nUniform Attention:")
    print(f"  Accuracies: {[f'{acc*100:.2f}%' for acc in uniform_accs]}")
    print(f"  Mean: {uniform_mean*100:.2f}% ± {uniform_std*100:.2f}%")

    # Paired comparison
    diff = np.array(learned_accs) - np.array(uniform_accs)
    print(f"\nDifference (Learned - Uniform):")
    print(f"  Per seed: {[f'{d*100:+.2f}%' for d in diff]}")
    print(f"  Mean difference: {np.mean(diff)*100:+.2f}%")

    if learned_mean > uniform_mean:
        print(f"\n✓ Learned attention performs better on average!")
    else:
        print(f"\n✗ Uniform attention performs better on average (unexpected)")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar plot
    x = np.arange(len(seeds))
    width = 0.35
    axes[0].bar(x - width/2, [acc*100 for acc in learned_accs], width,
                label='Learned Attention', color='#1f77b4', alpha=0.7)
    axes[0].bar(x + width/2, [acc*100 for acc in uniform_accs], width,
                label='Uniform Attention', color='#ff7f0e', alpha=0.7)
    axes[0].set_xlabel('Seed')
    axes[0].set_ylabel('Test Accuracy (%)')
    axes[0].set_title('Test Accuracy Comparison')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([str(s) for s in seeds])
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    # Average validation curves
    max_epochs = min(len(h['val_acc']) for h in learned_histories)
    learned_val = np.array([h['val_acc'][:max_epochs] for h in learned_histories])
    uniform_val = np.array([h['val_acc'][:max_epochs] for h in uniform_histories])

    axes[1].plot(range(max_epochs), learned_val.mean(axis=0),
                 label='Learned Attention', color='#1f77b4', linewidth=2)
    axes[1].plot(range(max_epochs), uniform_val.mean(axis=0),
                 label='Uniform Attention', color='#ff7f0e', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Validation Accuracy')
    axes[1].set_title('Average Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('quick_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to quick_comparison.png")

    print("="*80)


if __name__ == '__main__':
    main()
