"""
Main experiment script for stealthy model poisoning attacks
"""

import sys
import os
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import argparse
import copy
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Subset

from models.cnn_model import get_model
from attacks.baruch_attack import baruch_attack
from defenses.aggregation import fedavg_aggregate, trimmed_mean_aggregate, krum_aggregate
from utils.data_utils import load_data, create_non_iid_splits, FLClient, DEVICE
from utils.metrics import compute_stealthiness, evaluate_model
from utils.reproducibility import set_seed, get_git_info, save_results

# Experiment Configuration
NUM_CLIENTS = 20
NUM_ROUNDS = 30
NUM_MALICIOUS = 4
LOCAL_EPOCHS = 2
BATCH_SIZE = 32
LEARNING_RATE = 0.01  # client SGD; not yet tuned on the validation split
MOMENTUM = 0.9
MIN_SAMPLES_PER_CLIENT = 30
DIRICHLET_ALPHA = 0.5
ATTACK_STRENGTHS = [1.5, 2.0, 3.0, 5.0]
DEFENSE_ATTACK_STRENGTH = 3.0  # must be in ATTACK_STRENGTHS so its FedAvg run is shared
SEEDS = [0, 1, 2, 3, 4]

PLOTS_DIR = os.path.join(REPO_ROOT, 'plots')
RESULTS_DIR = os.path.join(REPO_ROOT, 'results')

# Each (attack_strength, defense) pair is run once per seed; z=1.0 is the no-attack baseline
BASELINE = (1.0, 'fedavg')
CONFIGS = ([BASELINE]
           + [(z, 'fedavg') for z in ATTACK_STRENGTHS]
           + [(DEFENSE_ATTACK_STRENGTH, 'trimmed_mean'), (DEFENSE_ATTACK_STRENGTH, 'krum')])

def run_experiment(attack_strength, defense='fedavg', seed=0, num_rounds=NUM_ROUNDS):
    """
    Run single FL experiment with given attack strength, defense and seed.

    Reseeding at the start makes the data partition, model initialisation and
    client shuffling identical for every configuration that shares a seed.
    """
    set_seed(seed)

    # Load data
    train_data, test_loader, num_classes = load_data()
    client_indices = create_non_iid_splits(train_data, NUM_CLIENTS, alpha=DIRICHLET_ALPHA,
                                           min_samples=MIN_SAMPLES_PER_CLIENT,
                                           rng=np.random.default_rng(seed))

    # Initialize clients
    base_model = get_model(num_classes, input_channels=3).to(DEVICE)
    clients = []
    for i, indices in enumerate(client_indices):
        if len(indices) >= MIN_SAMPLES_PER_CLIENT:
            subset = Subset(train_data, indices)
            clients.append(FLClient(i, subset, copy.deepcopy(base_model), DEVICE,
                                    BATCH_SIZE, LOCAL_EPOCHS, seed=seed * 1000 + i,
                                    lr=LEARNING_RATE, momentum=MOMENTUM))
        else:
            clients.append(None)

    active_clients = [c for c in clients if c is not None and c.has_data]
    num_valid_clients = len(active_clients)
    print(f"  Active clients: {num_valid_clients}")

    if num_valid_clients < 10:
        raise RuntimeError(f"Only {num_valid_clients} clients have enough data (seed {seed})")

    actual_malicious = min(NUM_MALICIOUS, max(2, num_valid_clients // 5))
    print(f"  Malicious clients: {actual_malicious}")

    # Initialize global model; the server keeps its weights on CPU, like the client updates
    global_model = copy.deepcopy(base_model)
    global_weights = {k: v.cpu().clone() for k, v in global_model.state_dict().items()}

    accuracies = []
    stealthiness_scores = []

    # Federated training rounds
    for round_idx in range(num_rounds):
        client_weights = []
        client_num_samples = []  # malicious clients report their true sample counts

        for i, client in enumerate(clients):
            if client is None or not client.has_data:
                continue

            # Local training
            client.set_weights(copy.deepcopy(global_weights))
            honest_weights = client.train()

            if honest_weights is None:
                continue
            client_num_samples.append(client.num_samples)

            # Apply attack if malicious
            is_malicious = (i < actual_malicious and attack_strength > 1.0)

            if is_malicious:
                poisoned = baruch_attack(global_weights, honest_weights, attack_strength)
                client_weights.append(poisoned)
                stealthiness_scores.append(compute_stealthiness(global_weights, poisoned))
            else:
                client_weights.append(honest_weights)

        if len(client_weights) == 0:
            continue

        # Aggregate updates
        if defense == 'trimmed_mean':
            global_weights = trimmed_mean_aggregate(client_weights)
        elif defense == 'krum':
            global_weights = krum_aggregate(client_weights, actual_malicious)
        else:  # fedavg
            global_weights = fedavg_aggregate(client_weights, client_num_samples)

        if global_weights is not None:
            global_model.load_state_dict(global_weights)
            acc = evaluate_model(global_model, test_loader, DEVICE)
            accuracies.append(acc)

            if (round_idx + 1) % 10 == 0:
                print(f"    Round {round_idx + 1}: Acc = {acc:.4f}")

    return {
        'attack_strength': attack_strength,
        'defense': defense,
        'seed': seed,
        'num_malicious': actual_malicious,
        'client_sizes': [len(idx) for idx in client_indices],
        'final_accuracy': accuracies[-1] if accuracies else 0,
        'stealthiness': float(np.mean(stealthiness_scores)) if stealthiness_scores else 0.0,
        'accuracies': accuracies
    }

def config_name(attack_strength, defense):
    return f'{defense}_z{attack_strength}'

def summarise(runs):
    """Mean and sample std over seeds of the final accuracy and stealthiness"""
    summary = {'num_seeds': len(runs), 'seeds': [r['seed'] for r in runs]}
    for metric in ['final_accuracy', 'stealthiness']:
        values = [r[metric] for r in runs]
        summary[f'{metric}_mean'] = float(np.mean(values))
        summary[f'{metric}_std'] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return summary

def plot_curve(ax, runs, label, style):
    """Mean accuracy per round with a ±1 std band over seeds"""
    curves = np.array([r['accuracies'] for r in runs])
    mean, std = curves.mean(axis=0), curves.std(axis=0)
    rounds = np.arange(1, curves.shape[1] + 1)
    line, = ax.plot(rounds, mean, style, label=label, linewidth=2)
    ax.fill_between(rounds, mean - std, mean + std, color=line.get_color(), alpha=0.2)

def plot_results(runs, stats):
    """Generate comprehensive results plots (mean ± std over seeds)"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    baseline = stats[BASELINE]
    baseline_label = (f'Baseline: {baseline["final_accuracy_mean"]:.3f} '
                      f'± {baseline["final_accuracy_std"]:.3f}')
    attack_stats = [stats[(z, 'fedavg')] for z in ATTACK_STRENGTHS]

    # Plot 1: Accuracy vs Attack Strength
    axes[0, 0].errorbar(ATTACK_STRENGTHS, [s['final_accuracy_mean'] for s in attack_stats],
                        yerr=[s['final_accuracy_std'] for s in attack_stats],
                        fmt='ro-', linewidth=2, markersize=8, capsize=4, label='FedAvg')
    axes[0, 0].axhline(y=baseline['final_accuracy_mean'], color='g', linestyle='--',
                       label=baseline_label)
    axes[0, 0].set_xlabel('Attack Strength (z)')
    axes[0, 0].set_ylabel('Final Test Accuracy')
    axes[0, 0].set_title('Model Accuracy vs Attack Strength')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Stealthiness (Lower = More Stealthy)
    axes[0, 1].errorbar(ATTACK_STRENGTHS, [s['stealthiness_mean'] for s in attack_stats],
                        yerr=[s['stealthiness_std'] for s in attack_stats],
                        fmt='bs-', linewidth=2, markersize=8, capsize=4)
    axes[0, 1].set_xlabel('Attack Strength (z)')
    axes[0, 1].set_ylabel('Attack Magnitude (L2 norm)')
    axes[0, 1].set_title('Stealthiness (Lower = Harder to Detect)')
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Defense Comparison
    defense_names = ['FedAvg (No Defense)', 'Trimmed Mean', 'Krum']
    defense_stats = [stats[(DEFENSE_ATTACK_STRENGTH, d)] for d in ['fedavg', 'trimmed_mean', 'krum']]
    defense_means = [s['final_accuracy_mean'] for s in defense_stats]
    defense_stds = [s['final_accuracy_std'] for s in defense_stats]
    bars = axes[1, 0].bar(defense_names, defense_means, yerr=defense_stds, capsize=6,
                          color=['red', 'blue', 'green'])
    axes[1, 0].axhline(y=baseline['final_accuracy_mean'], color='black', linestyle='--',
                       label=baseline_label)
    axes[1, 0].set_ylabel('Test Accuracy')
    axes[1, 0].set_title(f'Defense Effectiveness (Attack Strength = {DEFENSE_ATTACK_STRENGTH})')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    for bar, mean, std in zip(bars, defense_means, defense_stds):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01,
                       f'{mean:.3f}', ha='center', fontsize=9)

    # Plot 4: Training Progress
    for strength in ATTACK_STRENGTHS[:3]:
        plot_curve(axes[1, 1], runs[(strength, 'fedavg')], f'z={strength}', '-')
    plot_curve(axes[1, 1], runs[BASELINE], 'Baseline', 'g--')
    axes[1, 1].set_xlabel('Round')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_title('Training Progress Over Rounds')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(f'Stealthy Model Poisoning Attack Analysis (Baruch et al. 2019), '
                 f'mean ± std over {baseline["num_seeds"]} seeds', fontsize=14)
    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(PLOTS_DIR, 'experiment_results.png'), dpi=150, bbox_inches='tight')
    plt.close()

def print_summary(stats):
    def fmt(s):
        return f"{s['final_accuracy_mean']:.4f} ± {s['final_accuracy_std']:.4f}"

    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    print(f"Configuration: {NUM_CLIENTS} clients, {NUM_MALICIOUS} malicious, "
          f"seeds {stats[BASELINE]['seeds']}")
    print(f"Baseline Accuracy (No Attack): {fmt(stats[BASELINE])}")
    print("\nAttack Strength Results:")
    print("-" * 60)
    print(f"{'Strength':<10} {'Accuracy (mean ± std)':<24} {'Attack Magnitude':<20}")
    print("-" * 60)
    for z in ATTACK_STRENGTHS:
        s = stats[(z, 'fedavg')]
        print(f"z={z:<8} {fmt(s):<24} {s['stealthiness_mean']:.2f} ± {s['stealthiness_std']:.2f}")
    print(f"\nDefense Results (Attack Strength = {DEFENSE_ATTACK_STRENGTH}):")
    print(f"  FedAvg (No Defense):  {fmt(stats[(DEFENSE_ATTACK_STRENGTH, 'fedavg')])}")
    print(f"  Trimmed Mean:         {fmt(stats[(DEFENSE_ATTACK_STRENGTH, 'trimmed_mean')])}")
    print(f"  Krum:                 {fmt(stats[(DEFENSE_ATTACK_STRENGTH, 'krum')])}")
    print(f"\n✅ Results saved to {RESULTS_DIR} and {os.path.join(PLOTS_DIR, 'experiment_results.png')}")

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, nargs='+', default=SEEDS,
                        help='Random seeds; every configuration is run once per seed')
    parser.add_argument('--rounds', type=int, default=NUM_ROUNDS, help='Number of FL rounds')
    return parser.parse_args()

def main():
    # Windows consoles and redirected logs default to cp1252, which cannot encode ✅
    sys.stdout.reconfigure(encoding='utf-8')
    args = parse_args()
    config = {
        'num_clients': NUM_CLIENTS,
        'num_rounds': args.rounds,
        'num_malicious': NUM_MALICIOUS,
        'fedavg_weighting': 'client sample count (malicious clients report true counts)',
        'local_epochs': LOCAL_EPOCHS,
        'batch_size': BATCH_SIZE,
        'optimizer': {'name': 'sgd', 'lr': LEARNING_RATE, 'momentum': MOMENTUM,
                      'state': 'reset every round'},
        'min_samples_per_client': MIN_SAMPLES_PER_CLIENT,
        'dirichlet_alpha': DIRICHLET_ALPHA,
        'seeds': args.seeds,
        'git': get_git_info()
    }

    print("=" * 70)
    print("STEALTHY MODEL POISONING ATTACK EXPERIMENT")
    print(f"Clients: {NUM_CLIENTS}, Malicious: {NUM_MALICIOUS}, Rounds: {args.rounds}, "
          f"Seeds: {args.seeds}")
    print("=" * 70)

    runs = {cfg: [] for cfg in CONFIGS}
    for seed in args.seeds:
        for attack_strength, defense in CONFIGS:
            label = 'No attack' if (attack_strength, defense) == BASELINE else f'z={attack_strength}'
            print(f"\n[seed {seed}] {defense}, {label}")
            result = run_experiment(attack_strength, defense, seed, args.rounds)
            runs[(attack_strength, defense)].append(result)
            save_results(os.path.join(RESULTS_DIR, f'seed{seed}_{config_name(attack_strength, defense)}.json'),
                         {'config': config, **result})

    stats = {cfg: summarise(r) for cfg, r in runs.items()}
    save_results(os.path.join(RESULTS_DIR, 'summary.json'),
                 {'config': config, 'summary': {config_name(*cfg): s for cfg, s in stats.items()}})

    plot_results(runs, stats)
    print_summary(stats)
    print("\n✅ Experiment completed successfully!")

if __name__ == "__main__":
    main()
