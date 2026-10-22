"""
Main experiment script: model poisoning attacks vs. robust aggregation in FL
"""

import sys
import os
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import argparse
import copy
import json
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Subset

from models.cnn_model import get_model
from attacks import apply_attack
from defenses.aggregation import aggregate
from utils.data_utils import load_data, partition_data, FLClient, DEVICE
from utils.metrics import compute_stealthiness, evaluate, detection_rates
from utils.reproducibility import set_seed, get_git_info, get_environment_info, save_results

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
    target_class = attack['target_class'] if attack is not None else None

    # Load data
    train_data, val_loader, test_loader, num_classes = load_data()
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

    history = []  # per-round validation metrics and defense decisions
    stealthiness_scores = []

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
        kept_ids = None if accepted is None else [client_ids[j] for j in accepted]
        global_model.load_state_dict(global_weights)

        # Monitor on the validation split only; the test split is evaluated once, after training
        val = evaluate(global_model, val_loader, DEVICE, num_classes, target_class)
        history.append({
            'round': round_idx + 1,
            'val_accuracy': val['accuracy'],
            'val_balanced_accuracy': val['balanced_accuracy'],
            'val_nonfinite_samples': val['nonfinite_samples'],
            'kept_clients': kept_ids,
            'detection': detection_rates(kept_ids, client_ids, malicious_ids),
        })

        if (round_idx + 1) % 10 == 0:
            print(f"    Round {round_idx + 1}: Val acc = {val['accuracy']:.4f}, "
                  f"balanced = {val['balanced_accuracy']:.4f}")

    test = evaluate(global_model, test_loader, DEVICE, num_classes, target_class)
    print(f"  Test: acc = {test['accuracy']:.4f}, balanced = {test['balanced_accuracy']:.4f}")
    if test['nonfinite_samples']:
        print(f"  WARNING: {test['nonfinite_samples']} test samples had non-finite logits")

    return {
        'spec': spec,
        'seed': seed,
        'num_malicious': len(malicious_ids),
        'malicious_ids': sorted(malicious_ids),
        'client_sizes': [len(idx) for idx in client_indices],
        'stealthiness': float(np.mean(stealthiness_scores)) if stealthiness_scores else 0.0,
        'final_val': val,
        'test': test,
        'history': history,
    }

def mean_detection(run, key):
    """Mean of a detection rate over rounds, or None if the defense does not select clients"""
    values = [h['detection'][key] for h in run['history']
              if h['detection'] is not None and h['detection'][key] is not None]
    return float(np.mean(values)) if values else None

def mean_std(values):
    return {'mean': float(np.mean(values)),
            'std': float(np.std(values, ddof=1)) if len(values) > 1 else 0.0}

def summarise(runs):
    """Mean and sample std over seeds of the final test metrics, stealthiness and detection"""
    summary = {'num_seeds': len(runs), 'seeds': [r['seed'] for r in runs]}
    summary['test_accuracy'] = mean_std([r['test']['accuracy'] for r in runs])
    summary['test_balanced_accuracy'] = mean_std([r['test']['balanced_accuracy'] for r in runs])
    summary['test_per_class_recall_mean'] = np.nanmean(
        [[np.nan if v is None else v for v in r['test']['per_class_recall']] for r in runs],
        axis=0).tolist()
    summary['stealthiness'] = mean_std([r['stealthiness'] for r in runs])
    for key in ['tpr', 'fpr']:
        values = [mean_detection(r, key) for r in runs]
        summary[f'detection_{key}'] = (mean_std(values) if all(v is not None for v in values)
                                       else None)
    summary['test_nonfinite_samples'] = sum(r['test']['nonfinite_samples'] for r in runs)
    return summary

def plot_curve(ax, runs, label, style):
    """Mean validation accuracy per round with a ±1 std band over seeds"""
    curves = np.array([[h['val_accuracy'] for h in r['history']] for r in runs])
    mean, std = curves.mean(axis=0), curves.std(axis=0)
    rounds = np.arange(1, curves.shape[1] + 1)
    line, = ax.plot(rounds, mean, style, label=label, linewidth=2)
    ax.fill_between(rounds, mean - std, mean + std, color=line.get_color(), alpha=0.2)

def plot_results(runs, stats, plots_dir):
    """Generate comprehensive results plots (mean ± std over seeds)"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    baseline = stats[BASELINE['name']]
    baseline_acc = baseline['test_accuracy']
    baseline_label = f'Baseline: {baseline_acc["mean"]:.3f} ± {baseline_acc["std"]:.3f}'
    attack_stats = [stats[s['name']] for s in ATTACK_SWEEP]

    # Plot 1: Accuracy vs Attack Strength
    for metric, fmt, label in [('test_accuracy', 'ro-', 'FedAvg accuracy'),
                               ('test_balanced_accuracy', 'm^--', 'FedAvg balanced accuracy')]:
        axes[0, 0].errorbar(ATTACK_STRENGTHS, [s[metric]['mean'] for s in attack_stats],
                            yerr=[s[metric]['std'] for s in attack_stats],
                            fmt=fmt, linewidth=2, markersize=8, capsize=4, label=label)
    axes[0, 0].axhline(y=baseline_acc['mean'], color='g', linestyle='--', label=baseline_label)
    axes[0, 0].set_xlabel('Attack Strength (z)')
    axes[0, 0].set_ylabel('Final Test Accuracy')
    axes[0, 0].set_title('Model Accuracy vs Attack Strength')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Stealthiness (Lower = More Stealthy)
    axes[0, 1].errorbar(ATTACK_STRENGTHS, [s['stealthiness']['mean'] for s in attack_stats],
                        yerr=[s['stealthiness']['std'] for s in attack_stats],
                        fmt='bs-', linewidth=2, markersize=8, capsize=4)
    axes[0, 1].set_xlabel('Attack Strength (z)')
    axes[0, 1].set_ylabel('Attack Magnitude (L2 norm)')
    axes[0, 1].set_title('Stealthiness (Lower = Harder to Detect)')
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Defense Comparison
    defense_names = [DEFENSE_LABELS[s['defense']['name']] for s in DEFENSE_COMPARISON]
    defense_stats = [stats[s['name']] for s in DEFENSE_COMPARISON]
    defense_means = [s['test_accuracy']['mean'] for s in defense_stats]
    defense_stds = [s['test_accuracy']['std'] for s in defense_stats]
    bars = axes[1, 0].bar(defense_names, defense_means, yerr=defense_stds, capsize=6,
                          color=['red', 'blue', 'green'])
    axes[1, 0].axhline(y=baseline_acc['mean'], color='black', linestyle='--',
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
    axes[1, 1].set_ylabel('Validation Accuracy')
    axes[1, 1].set_title('Training Progress Over Rounds (validation split)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(f'Naive Reversed/Scaled Poisoning Attack, '
                 f'mean ± std over {baseline["num_seeds"]} seeds', fontsize=14)
    plt.tight_layout()
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, 'experiment_results.png'), dpi=150, bbox_inches='tight')
    plt.close()

def print_summary(stats, results_dir, plots_dir):
    def fmt(m):
        return f"{m['mean']:.4f} ± {m['std']:.4f}"

    baseline = stats[BASELINE['name']]
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY (test split, mean ± std over seeds)")
    print("=" * 70)
    print(f"Configuration: {NUM_CLIENTS} clients, {NUM_MALICIOUS} malicious, seeds {baseline['seeds']}")
    print(f"Baseline (No Attack): acc {fmt(baseline['test_accuracy'])}, "
          f"balanced {fmt(baseline['test_balanced_accuracy'])}")
    print("\nAttack Strength Results (FedAvg):")
    print("-" * 70)
    print(f"{'Strength':<10} {'Accuracy':<20} {'Balanced accuracy':<20} {'Attack magnitude':<18}")
    print("-" * 70)
    for spec, z in zip(ATTACK_SWEEP, ATTACK_STRENGTHS):
        s = stats[spec['name']]
        print(f"z={z:<8} {fmt(s['test_accuracy']):<20} {fmt(s['test_balanced_accuracy']):<20} "
              f"{s['stealthiness']['mean']:.2f} ± {s['stealthiness']['std']:.2f}")
    print(f"\nDefense Results (Attack Strength = {DEFENSE_ATTACK_STRENGTH}):")
    for spec in DEFENSE_COMPARISON:
        s = stats[spec['name']]
        label = DEFENSE_LABELS[spec['defense']['name']] + ':'
        line = (f"  {label:<22}acc {fmt(s['test_accuracy'])}, "
                f"balanced {fmt(s['test_balanced_accuracy'])}")
        if s['detection_tpr'] is not None:
            line += (f", detection TPR {fmt(s['detection_tpr'])}"
                     f", FPR {fmt(s['detection_fpr'])}")
        print(line)
    nonfinite = sum(s['test_nonfinite_samples'] for s in stats.values())
    if nonfinite:
        print(f"\nWARNING: {nonfinite} test predictions came from non-finite logits")
    print(f"\n✅ Results saved to {results_dir} and {os.path.join(plots_dir, 'experiment_results.png')}")

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
        'evaluation': 'validation split every round; test split once after the final round',
        'detection': 'clients a defense did not keep count as flagged (Krum keeps one)',
        'seeds': args.seeds,
        'environment': get_environment_info(DEVICE),
        'git': get_git_info()
    }

# Config keys that may legitimately differ between runs that are summarised together
RUN_SPECIFIC_CONFIG_KEYS = {'seeds', 'git', 'environment'}

def result_path(results_dir, seed, spec):
    return os.path.join(results_dir, f"seed{seed}_{spec['name']}.json")

def load_result(path, spec, seed, config):
    """
    Load a saved run for resuming. Refuses results produced with different
    experiment settings, so runs are never silently mixed; a different commit
    or device only triggers a warning.
    """
    with open(path) as f:
        result = json.load(f)
    if result['spec'] != spec or result['seed'] != seed:
        raise ValueError(f"{path} does not contain seed {seed} of {spec['name']}")
    saved = result['config']
    differing = sorted(k for k in set(saved) | set(config)
                       if k not in RUN_SPECIFIC_CONFIG_KEYS and saved.get(k) != config.get(k))
    if differing:
        raise ValueError(f"{path} was produced with different settings ({', '.join(differing)}); "
                         f"use another --results-dir or delete it")
    if saved['git']['commit'] != config['git']['commit']:
        print(f"  WARNING: {os.path.basename(path)} comes from commit {saved['git']['commit'][:7]}")
    if saved['environment']['device'] != config['environment']['device']:
        print(f"  WARNING: {os.path.basename(path)} was run on {saved['environment']['device']}")
    return result

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, nargs='+', default=SEEDS,
                        help='Random seeds; every configuration is run once per seed')
    parser.add_argument('--rounds', type=int, default=NUM_ROUNDS, help='Number of FL rounds')
    parser.add_argument('--results-dir', default=RESULTS_DIR,
                        help='Where per-run JSON files and summary.json are written; runs whose '
                             'file already exists here are skipped (resume)')
    parser.add_argument('--plots-dir', default=PLOTS_DIR, help='Where the summary plot is written')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--skip-summary', action='store_true',
                      help='Only run missing experiments; do not write summary.json or the plot')
    mode.add_argument('--summary-only', action='store_true',
                      help='Do not train; build summary.json and the plot from saved runs '
                           '(fails if any run is missing)')
    return parser.parse_args()

def main():
    # Windows consoles and redirected logs default to cp1252, which cannot encode ✅
    sys.stdout.reconfigure(encoding='utf-8')
    args = parse_args()
    config = build_config(args)

    print("=" * 70)
    print("MODEL POISONING ATTACK EXPERIMENT")
    print(f"Clients: {NUM_CLIENTS}, Malicious: {NUM_MALICIOUS}, Rounds: {args.rounds}, "
          f"Seeds: {args.seeds}, Device: {config['environment']['device']}")
    print(f"Results: {args.results_dir}")
    print("=" * 70)

    runs = {spec['name']: [] for spec in CONFIGS}
    for seed in args.seeds:
        for spec in CONFIGS:
            path = result_path(args.results_dir, seed, spec)
            if os.path.exists(path):
                print(f"\n[seed {seed}] {spec['name']}: already done, skipping")
                result = load_result(path, spec, seed, config)
            elif args.summary_only:
                raise FileNotFoundError(f"Missing {path}; run without --summary-only first")
            else:
                print(f"\n[seed {seed}] {spec['name']}")
                result = run_experiment(spec, seed, args.rounds)
                save_results(path, {'config': config, **result})
            runs[spec['name']].append(result)

    if args.skip_summary:
        print("\n✅ Runs completed (summary skipped)")
        return

    stats = {name: summarise(r) for name, r in runs.items()}
    save_results(os.path.join(args.results_dir, 'summary.json'), {'config': config, 'summary': stats})

    plot_results(runs, stats, args.plots_dir)
    print_summary(stats, args.results_dir, args.plots_dir)
    print("\n✅ Experiment completed successfully!")

if __name__ == "__main__":
    main()
