"""
Rigorous comparison of Learned Attention vs Uniform Attention in GAT

This script runs multiple experiments with different seeds to get statistically
meaningful comparison between learned and uniform attention.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from gat_pytorch import GAT, masked_softmax_cross_entropy, masked_accuracy, load_and_prepare_data
from experiments import GATVariant, run_experiment


def run_multiple_seeds(attention_type, seeds=[42, 123, 456, 789, 1024]):
    """
    Run experiments with multiple seeds and collect results
    """
    results = []
    print(f"\n{'='*60}")
    print(f"Running {attention_type.upper()} ATTENTION with {len(seeds)} seeds")
    print(f"{'='*60}")

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        torch.manual_seed(seed)
        np.random.seed(seed)

        history, test_acc = run_experiment(attention_type, seed=seed)
        results.append({
            'seed': seed,
            'test_acc': test_acc,
            'history': history
        })

    return results


def analyze_results(learned_results, uniform_results):
    """
    Analyze and compare results from multiple runs
    """
    learned_accs = [r['test_acc'] for r in learned_results]
    uniform_accs = [r['test_acc'] for r in uniform_results]

    # Statistics
    learned_mean = np.mean(learned_accs)
    learned_std = np.std(learned_accs)
    uniform_mean = np.mean(uniform_accs)
    uniform_std = np.std(uniform_accs)

    print("\n" + "="*80)
    print("STATISTICAL COMPARISON (Multiple Seeds)")
    print("="*80)
    print(f"\nLearned Attention:")
    print(f"  Test accuracies: {[f'{acc*100:.2f}%' for acc in learned_accs]}")
    print(f"  Mean: {learned_mean*100:.2f}% ± {learned_std*100:.2f}%")

    print(f"\nUniform Attention:")
    print(f"  Test accuracies: {[f'{acc*100:.2f}%' for acc in uniform_accs]}")
    print(f"  Mean: {uniform_mean*100:.2f}% ± {uniform_std*100:.2f}%")

    # Paired t-test
    from scipy import stats
    t_stat, p_value = stats.ttest_rel(learned_accs, uniform_accs)

    print(f"\nPaired t-test:")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.4f}")

    if p_value < 0.05:
        if learned_mean > uniform_mean:
            print(f"  ✓ Learned attention is SIGNIFICANTLY better (p < 0.05)")
        else:
            print(f"  ✗ Uniform attention is SIGNIFICANTLY better (p < 0.05)")
    else:
        print(f"  = No significant difference (p >= 0.05)")

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((learned_std**2 + uniform_std**2) / 2)
    cohens_d = (learned_mean - uniform_mean) / pooled_std
    print(f"\nEffect size (Cohen's d): {cohens_d:.4f}")

    return {
        'learned_mean': learned_mean,
        'learned_std': learned_std,
        'uniform_mean': uniform_mean,
        'uniform_std': uniform_std,
        't_stat': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'learned_accs': learned_accs,
        'uniform_accs': uniform_accs
    }


def plot_comparison(stats, learned_results, uniform_results, save_path='attention_comparison_rigorous.png'):
    """
    Plot comparison of learned vs uniform attention
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Bar plot with error bars
    ax = axes[0, 0]
    methods = ['Learned\nAttention', 'Uniform\nAttention']
    means = [stats['learned_mean'] * 100, stats['uniform_mean'] * 100]
    stds = [stats['learned_std'] * 100, stats['uniform_std'] * 100]

    bars = ax.bar(methods, means, yerr=stds, capsize=10,
                  color=['#1f77b4', '#ff7f0e'], alpha=0.7, edgecolor='black')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Test Accuracy Comparison (Mean ± Std)')
    ax.set_ylim([min(means) - max(stds) - 2, max(means) + max(stds) + 2])
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.5,
                f'{mean:.2f}%', ha='center', va='bottom', fontweight='bold')

    # 2. Box plot
    ax = axes[0, 1]
    data = [stats['learned_accs'], stats['uniform_accs']]
    bp = ax.boxplot(data, labels=['Learned Attention', 'Uniform Attention'],
                    patch_artist=True)
    bp['boxes'][0].set_facecolor('#1f77b4')
    bp['boxes'][1].set_facecolor('#ff7f0e')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Distribution of Test Accuracies')
    ax.grid(True, alpha=0.3, axis='y')

    # 3. Individual runs comparison
    ax = axes[1, 0]
    seeds = range(len(stats['learned_accs']))
    x = np.arange(len(seeds))
    width = 0.35

    ax.bar(x - width/2, [acc*100 for acc in stats['learned_accs']], width,
           label='Learned Attention', color='#1f77b4', alpha=0.7)
    ax.bar(x + width/2, [acc*100 for acc in stats['uniform_accs']], width,
           label='Uniform Attention', color='#ff7f0e', alpha=0.7)

    ax.set_xlabel('Run Index')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Test Accuracy by Run')
    ax.set_xticks(x)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 4. Training curves (average across seeds)
    ax = axes[1, 1]

    # Average training curves
    max_epochs = min(len(r['history']['val_acc']) for r in learned_results)

    learned_val_accs = np.array([r['history']['val_acc'][:max_epochs] for r in learned_results])
    uniform_val_accs = np.array([r['history']['val_acc'][:max_epochs] for r in uniform_results])

    learned_mean_curve = learned_val_accs.mean(axis=0)
    learned_std_curve = learned_val_accs.std(axis=0)
    uniform_mean_curve = uniform_val_accs.mean(axis=0)
    uniform_std_curve = uniform_val_accs.std(axis=0)

    epochs = range(max_epochs)

    ax.plot(epochs, learned_mean_curve, label='Learned Attention', color='#1f77b4', linewidth=2)
    ax.fill_between(epochs, learned_mean_curve - learned_std_curve,
                    learned_mean_curve + learned_std_curve, alpha=0.3, color='#1f77b4')

    ax.plot(epochs, uniform_mean_curve, label='Uniform Attention', color='#ff7f0e', linewidth=2)
    ax.fill_between(epochs, uniform_mean_curve - uniform_std_curve,
                    uniform_mean_curve + uniform_std_curve, alpha=0.3, color='#ff7f0e')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Accuracy')
    ax.set_title('Average Validation Accuracy Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nComparison plot saved to {save_path}")


def main():
    """Main function"""
    # Use multiple seeds for statistical significance
    seeds = [42, 123, 456, 789, 1024]

    print("="*80)
    print("RIGOROUS COMPARISON: LEARNED vs UNIFORM ATTENTION")
    print("="*80)
    print(f"Running {len(seeds)} experiments for each attention type...")

    # Run experiments
    learned_results = run_multiple_seeds('learned', seeds)
    uniform_results = run_multiple_seeds('uniform', seeds)

    # Analyze results
    stats = analyze_results(learned_results, uniform_results)

    # Plot comparison
    plot_comparison(stats, learned_results, uniform_results)

    print("\n" + "="*80)
    print("CONCLUSION:")
    print("="*80)

    if stats['p_value'] < 0.05:
        if stats['learned_mean'] > stats['uniform_mean']:
            improvement = (stats['learned_mean'] - stats['uniform_mean']) / stats['uniform_mean'] * 100
            print(f"✓ Learned attention is significantly better than uniform attention")
            print(f"  Relative improvement: {improvement:+.2f}%")
            print(f"  This validates that the attention mechanism learns meaningful patterns!")
        else:
            print(f"✗ Uniform attention is significantly better (unexpected result)")
            print(f"  This may indicate issues with the implementation or hyperparameters.")
    else:
        print(f"= No statistically significant difference found")
        print(f"  Both methods perform similarly on this dataset.")
        print(f"  This could mean:")
        print(f"    1. The graph structure is already informative enough")
        print(f"    2. The attention mechanism needs more capacity")
        print(f"    3. The dataset is too small to show the difference")

    print("="*80)


if __name__ == '__main__':
    main()
