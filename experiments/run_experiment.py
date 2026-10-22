"""
Main experiment script: model poisoning attacks vs. robust aggregation in FL
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
from attacks import apply_attack
from defenses.aggregation import aggregate
from utils.data_utils import load_data, partition_data, FLClient, DEVICE
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
PARTITION = {'type': 'dirichlet', 'alpha': 0.5, 'min_samples': 30}
ATTACK_STRENGTHS = [1.5, 2.0, 3.0, 5.0]
DEFENSE_ATTACK_STRENGTH = 3.0  # must be in ATTACK_STRENGTHS so its FedAvg run is shared
SEEDS = [0, 1, 2, 3, 4]

PLOTS_DIR = os.path.join(REPO_ROOT, 'plots')
RESULTS_DIR = os.path.join(REPO_ROOT, 'results')

def make_spec(defense, scale=None, **defense_params):
    """
    One experiment configuration. attack is None for the no-attack baseline;
    target_class is reserved for targeted attacks (None = untargeted).
    """
    attack = None
    name = f'{defense}_no_attack'
    if scale is not None:
        attack = {'name': 'reversed_scaled', 'params': {'scale': scale}, 'target_class': None}
        name = f'{defense}_reversed_scaled_z{scale}'
    return {'name': name, 'attack': attack,
            'defense': {'name': defense, 'params': defense_params}}

BASELINE = make_spec('fedavg')
ATTACK_SWEEP = [make_spec('fedavg', z) for z in ATTACK_STRENGTHS]
DEFENSE_COMPARISON = [make_spec('fedavg', DEFENSE_ATTACK_STRENGTH),
                      make_spec('trimmed_mean', DEFENSE_ATTACK_STRENGTH, trim_ratio=0.2),
                      make_spec('krum', DEFENSE_ATTACK_STRENGTH)]  # Krum f = true #malicious
DEFENSE_LABELS = {'fedavg': 'FedAvg (No Defense)', 'trimmed_mean': 'Trimmed Mean', 'krum': 'Krum'}

# Every distinct spec is run once per seed (FedAvg at z=3 is shared by sweep and comparison)
CONFIGS = list({s['name']: s for s in [BASELINE] + ATTACK_SWEEP + DEFENSE_COMPARISON}.values())

def run_experiment(spec, seed=0, num_rounds=NUM_ROUNDS):
    """
    Run single FL experiment for one spec and seed.

    Reseeding at the start makes the data partition, model initialisation and
    client shuffling identical for every configuration that shares a seed.
    """
    set_seed(seed)
    attack, defense = spec['attack'], spec['defense']

    # Load data
    train_data, test_loader, num_classes = load_data()
    client_indices = partition_data(train_data, NUM_CLIENTS, PARTITION,
                                    rng=np.random.default_rng(seed))

    # Initialize clients
    base_model = get_model(num_classes, input_channels=3).to(DEVICE)
    clients = []
    for i, indices in enumerate(client_indices):
        if len(indices) >= PARTITION['min_samples']:
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
    malicious_ids = set(range(actual_malicious)) if attack is not None else set()
    print(f"  Malicious clients: {sorted(malicious_ids)}")

    # Initialize global model; the server keeps its weights on CPU, like the client updates
    global_model = copy.deepcopy(base_model)
    global_weights = {k: v.cpu().clone() for k, v in global_model.state_dict().items()}

    accuracies = []
    stealthiness_scores = []
    accepted_per_round = []  # client ids each defense kept (None if it keeps no whole clients)

    # Federated training rounds
    for round_idx in range(num_rounds):
        client_weights = []
        client_num_samples = []  # malicious clients report their true sample counts
        client_ids = []

        for client in active_clients:
            # Local training
            client.set_weights(copy.deepcopy(global_weights))
            honest_weights = client.train()

            if honest_weights is None:
                continue
            client_num_samples.append(client.num_samples)
            client_ids.append(client.client_id)

            if client.client_id in malicious_ids:
                poisoned = apply_attack(attack['name'], global_weights, honest_weights,
                                        **attack['params'])
                client_weights.append(poisoned)
                stealthiness_scores.append(compute_stealthiness(global_weights, poisoned))
            else:
                client_weights.append(honest_weights)

        if len(client_weights) == 0:
            continue

        # Aggregate updates; defenses that need f assume the configured number of attackers
        global_weights, accepted = aggregate(defense['name'], client_weights, client_num_samples,
                                             actual_malicious, **defense['params'])
        accepted_per_round.append(None if accepted is None else [client_ids[j] for j in accepted])

        global_model.load_state_dict(global_weights)
        acc = evaluate_model(global_model, test_loader, DEVICE)
        accuracies.append(acc)

        if (round_idx + 1) % 10 == 0:
            print(f"    Round {round_idx + 1}: Acc = {acc:.4f}")

    return {
        'spec': spec,
        'seed': seed,
        'num_malicious': len(malicious_ids),
        'malicious_ids': sorted(malicious_ids),
        'client_sizes': [len(idx) for idx in client_indices],
        'final_accuracy': accuracies[-1] if accuracies else 0,
        'stealthiness': float(np.mean(stealthiness_scores)) if stealthiness_scores else 0.0,
        'accuracies': accuracies,
        'accepted_clients': accepted_per_round
    }

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

    baseline = stats[BASELINE['name']]
    baseline_label = (f'Baseline: {baseline["final_accuracy_mean"]:.3f} '
                      f'± {baseline["final_accuracy_std"]:.3f}')
    attack_stats = [stats[s['name']] for s in ATTACK_SWEEP]

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
    defense_names = [DEFENSE_LABELS[s['defense']['name']] for s in DEFENSE_COMPARISON]
    defense_stats = [stats[s['name']] for s in DEFENSE_COMPARISON]
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
    for s, strength in zip(ATTACK_SWEEP[:3], ATTACK_STRENGTHS[:3]):
        plot_curve(axes[1, 1], runs[s['name']], f'z={strength}', '-')
    plot_curve(axes[1, 1], runs[BASELINE['name']], 'Baseline', 'g--')
    axes[1, 1].set_xlabel('Round')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_title('Training Progress Over Rounds')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(f'Naive Reversed/Scaled Poisoning Attack, '
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
          f"seeds {stats[BASELINE['name']]['seeds']}")
    print(f"Baseline Accuracy (No Attack): {fmt(stats[BASELINE['name']])}")
    print("\nAttack Strength Results:")
    print("-" * 60)
    print(f"{'Strength':<10} {'Accuracy (mean ± std)':<24} {'Attack Magnitude':<20}")
    print("-" * 60)
    for spec, z in zip(ATTACK_SWEEP, ATTACK_STRENGTHS):
        s = stats[spec['name']]
        print(f"z={z:<8} {fmt(s):<24} {s['stealthiness_mean']:.2f} ± {s['stealthiness_std']:.2f}")
    print(f"\nDefense Results (Attack Strength = {DEFENSE_ATTACK_STRENGTH}):")
    for spec in DEFENSE_COMPARISON:
        label = DEFENSE_LABELS[spec['defense']['name']] + ':'
        print(f"  {label:<22}{fmt(stats[spec['name']])}")
    print(f"\n✅ Results saved to {RESULTS_DIR} and {os.path.join(PLOTS_DIR, 'experiment_results.png')}")

def build_config(args):
    """Settings shared by every run, stored alongside each result"""
    return {
        'num_clients': NUM_CLIENTS,
        'num_rounds': args.rounds,
        'num_malicious': NUM_MALICIOUS,
        'malicious_clients': 'client ids 0..num_malicious-1, attacking every round',
        'fedavg_weighting': 'client sample count (malicious clients report true counts)',
        'local_epochs': LOCAL_EPOCHS,
        'batch_size': BATCH_SIZE,
        'optimizer': {'name': 'sgd', 'lr': LEARNING_RATE, 'momentum': MOMENTUM,
                      'state': 'reset every round'},
        'model': {'name': 'MedicalCNN', 'norm': 'GroupNorm(8)'},
        'partition': PARTITION,
        'seeds': args.seeds,
        'git': get_git_info()
    }

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
    config = build_config(args)

    print("=" * 70)
    print("MODEL POISONING ATTACK EXPERIMENT")
    print(f"Clients: {NUM_CLIENTS}, Malicious: {NUM_MALICIOUS}, Rounds: {args.rounds}, "
          f"Seeds: {args.seeds}")
    print("=" * 70)

    runs = {spec['name']: [] for spec in CONFIGS}
    for seed in args.seeds:
        for spec in CONFIGS:
            print(f"\n[seed {seed}] {spec['name']}")
            result = run_experiment(spec, seed, args.rounds)
            runs[spec['name']].append(result)
            save_results(os.path.join(RESULTS_DIR, f"seed{seed}_{spec['name']}.json"),
                         {'config': config, **result})

    stats = {name: summarise(r) for name, r in runs.items()}
    save_results(os.path.join(RESULTS_DIR, 'summary.json'), {'config': config, 'summary': stats})

    plot_results(runs, stats)
    print_summary(stats)
    print("\n✅ Experiment completed successfully!")

if __name__ == "__main__":
    main()
