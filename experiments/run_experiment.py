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
                      make_spec('krum', DEFENSE_ATTACK_STRENGTH),  # Krum f = true #malicious
                      make_spec('multi_krum', DEFENSE_ATTACK_STRENGTH)]  # m = n - f
# Each defense also runs without attack, to separate the defense's own cost from the attack's effect
DEFENSE_CONTROLS = [BASELINE,
                    make_spec('trimmed_mean', trim_ratio=0.2),
                    make_spec('krum'),
                    make_spec('multi_krum')]
DEFENSE_PAIRS = list(zip(DEFENSE_CONTROLS, DEFENSE_COMPARISON))  # (no attack, under attack)
DEFENSE_LABELS = {'fedavg': 'FedAvg (No Defense)', 'trimmed_mean': 'Trimmed Mean', 'krum': 'Krum',
                  'multi_krum': 'Multi-Krum'}

# Every distinct spec is run once per seed (FedAvg at z=3 is shared by sweep and comparison)
CONFIGS = list({s['name']: s for s in [BASELINE] + ATTACK_SWEEP + DEFENSE_COMPARISON
                + DEFENSE_CONTROLS}.values())

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
    # Planned for the next phase: seeded random attackers per seed (docs/planned_changes.md)
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

def run_status(run):
    """
    'diverged': the final model gave non-finite logits on some test sample;
    'degenerate': finite, but every test prediction falls in a single class;
    'valid': otherwise. Only valid runs enter the means.
    """
    if run['test']['nonfinite_samples'] > 0:
        return 'diverged'
    predicted_classes = np.count_nonzero(np.asarray(run['test']['confusion_matrix']).sum(axis=0))
    return 'degenerate' if predicted_classes <= 1 else 'valid'

def first_nonfinite_round(run):
    """First round with non-finite validation outputs, or None if none were seen on validation"""
    return next((h['round'] for h in run['history'] if h['val_nonfinite_samples'] > 0), None)

def attacker_data_share(run):
    """
    Attackers' share of the training data held by participating clients, which is also
    their total weight in sample-weighted FedAvg; None when the run has no attackers.
    """
    if not run['malicious_ids']:
        return None
    sizes = run['client_sizes']
    participating = [n for n in sizes if n >= PARTITION['min_samples']]
    return sum(sizes[i] for i in run['malicious_ids']) / sum(participating)

def per_seed_row(run, status):
    return {
        'seed': run['seed'],
        'status': status,
        'first_nonfinite_round': first_nonfinite_round(run),
        'attacker_data_share': attacker_data_share(run),
        'test_accuracy': run['test']['accuracy'],
        'test_balanced_accuracy': run['test']['balanced_accuracy'],
        'stealthiness': run['stealthiness'],
    }

def summarise(runs):
    """
    Mean and sample std over seeds of the final test metrics, stealthiness and detection.
    Diverged and degenerate runs are counted, listed and excluded from every mean;
    a metric is None when no run is valid.
    """
    statuses = [run_status(r) for r in runs]
    valid = [r for r, status in zip(runs, statuses) if status == 'valid']
    summary = {'num_seeds': len(runs), 'seeds': [r['seed'] for r in runs]}
    for status in ['valid', 'diverged', 'degenerate']:
        seeds = [r['seed'] for r, st in zip(runs, statuses) if st == status]
        summary[f'num_{status}'] = len(seeds)
        summary[f'{status}_seeds'] = seeds
    summary['per_seed'] = [per_seed_row(r, status) for r, status in zip(runs, statuses)]

    def stat(values):
        return mean_std(values) if values else None
    summary['test_accuracy'] = stat([r['test']['accuracy'] for r in valid])
    summary['test_balanced_accuracy'] = stat([r['test']['balanced_accuracy'] for r in valid])
    summary['test_per_class_recall_mean'] = (np.nanmean(
        [[np.nan if v is None else v for v in r['test']['per_class_recall']] for r in valid],
        axis=0).tolist() if valid else None)
    summary['stealthiness'] = stat([r['stealthiness'] for r in valid])
    for key in ['tpr', 'fpr']:
        values = [mean_detection(r, key) for r in valid]
        summary[f'detection_{key}'] = (mean_std(values) if values and all(v is not None for v in values)
                                       else None)
    summary['test_nonfinite_samples'] = sum(r['test']['nonfinite_samples'] for r in runs)
    return summary

def counts_label(s):
    """Seeds behind a mean, e.g. '[5/5 valid]' or '[2/5 valid: 3 diverged]'"""
    excluded = [f"{s[f'num_{k}']} {k}" for k in ('diverged', 'degenerate') if s[f'num_{k}']]
    return f"[{s['num_valid']}/{s['num_seeds']} valid{': ' + ', '.join(excluded) if excluded else ''}]"

def format_metric(m, s, digits=4):
    """A mean ± std with the counts it was computed over; '—' when no run is valid"""
    value = '—' if m is None else f"{m['mean']:.{digits}f} ± {m['std']:.{digits}f}"
    return f"{value} {counts_label(s)}"

def mean_or_nan(m, key='mean'):
    return np.nan if m is None else m[key]

def plot_curve(ax, runs, label, style):
    """Mean validation accuracy per round with a ±1 std band over the valid seeds"""
    valid = [r for r in runs if run_status(r) == 'valid']
    label = f"{label} [{len(valid)}/{len(runs)} valid]"
    if not valid:
        ax.plot([], [], style, label=label + ' - not shown')
        return
    curves = np.array([[h['val_accuracy'] for h in r['history']] for r in valid])
    mean, std = curves.mean(axis=0), curves.std(axis=0)
    rounds = np.arange(1, curves.shape[1] + 1)
    line, = ax.plot(rounds, mean, style, label=label, linewidth=2)
    ax.fill_between(rounds, mean - std, mean + std, color=line.get_color(), alpha=0.2)

def annotate_counts(ax, xs, stat_list):
    """Write 'v/n valid' under each x position, so no mean is shown without its seed count"""
    # Fix the x range: if only some points have valid seeds, autoscaling would zoom onto
    # them and push the other labels far outside the axes
    margin = 0.1 * (max(xs) - min(xs)) or 0.5
    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_xticks(xs)
    for x, s in zip(xs, stat_list):
        ax.annotate(f"{s['num_valid']}/{s['num_seeds']} valid", xy=(x, 0.02),
                    xycoords=('data', 'axes fraction'), ha='center', fontsize=8,
                    color='black' if s['num_valid'] == s['num_seeds'] else 'red')

def plot_results(runs, stats, plots_dir):
    """Generate comprehensive results plots (mean ± std over seeds)"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    baseline = stats[BASELINE['name']]
    baseline_acc = baseline['test_accuracy']
    baseline_label = f"Baseline: {format_metric(baseline_acc, baseline, digits=3)}"
    attack_stats = [stats[s['name']] for s in ATTACK_SWEEP]

    # Plot 1: Accuracy vs Attack Strength (valid seeds only; counts under each point)
    for metric, fmt, label in [('test_accuracy', 'ro-', 'FedAvg accuracy'),
                               ('test_balanced_accuracy', 'm^--', 'FedAvg balanced accuracy')]:
        axes[0, 0].errorbar(ATTACK_STRENGTHS, [mean_or_nan(s[metric]) for s in attack_stats],
                            yerr=[mean_or_nan(s[metric], 'std') for s in attack_stats],
                            fmt=fmt, linewidth=2, markersize=8, capsize=4, label=label)
    if baseline_acc is not None:
        axes[0, 0].axhline(y=baseline_acc['mean'], color='g', linestyle='--', label=baseline_label)
    annotate_counts(axes[0, 0], ATTACK_STRENGTHS, attack_stats)
    axes[0, 0].set_xlabel('Attack Strength (z)')
    axes[0, 0].set_ylabel('Final Test Accuracy')
    axes[0, 0].set_title('Model Accuracy vs Attack Strength')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Stealthiness (Lower = More Stealthy)
    axes[0, 1].errorbar(ATTACK_STRENGTHS, [mean_or_nan(s['stealthiness']) for s in attack_stats],
                        yerr=[mean_or_nan(s['stealthiness'], 'std') for s in attack_stats],
                        fmt='bs-', linewidth=2, markersize=8, capsize=4)
    annotate_counts(axes[0, 1], ATTACK_STRENGTHS, attack_stats)
    axes[0, 1].set_xlabel('Attack Strength (z)')
    axes[0, 1].set_ylabel('Attack Magnitude (L2 norm)')
    axes[0, 1].set_title('Stealthiness (Lower = Harder to Detect)')
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Defense Comparison, each defense without and with attack
    x = np.arange(len(DEFENSE_PAIRS))
    width = 0.38
    for offset, idx, label, color in [(-width / 2, 0, 'No attack', 'tab:green'),
                                      (width / 2, 1, f'Attack z={DEFENSE_ATTACK_STRENGTH}', 'tab:red')]:
        pair_stats = [stats[pair[idx]['name']] for pair in DEFENSE_PAIRS]
        accs = [s['test_accuracy'] for s in pair_stats]
        bars = axes[1, 0].bar(x + offset, [mean_or_nan(a) for a in accs], width,
                              yerr=[mean_or_nan(a, 'std') for a in accs], capsize=4, color=color,
                              label=label)
        for bar, a, s in zip(bars, accs, pair_stats):
            counts = f"{s['num_valid']}/{s['num_seeds']}"
            text = f"{counts} valid" if a is None else f"{a['mean']:.3f}\n[{counts}]"
            height = 0.0 if a is None else a['mean'] + a['std']
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, height + 0.01, text, ha='center',
                            fontsize=8, color='black' if s['num_valid'] == s['num_seeds'] else 'red')
    axes[1, 0].set_xticks(x, [DEFENSE_LABELS[attacked['defense']['name']]
                              for _, attacked in DEFENSE_PAIRS])
    axes[1, 0].set_ylim(0, 1.15)  # room for the value/count labels above the bars
    axes[1, 0].set_ylabel('Test Accuracy')
    axes[1, 0].set_title('Defense Effectiveness (with and without attack)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Training Progress
    for s, strength in zip(ATTACK_SWEEP[:3], ATTACK_STRENGTHS[:3]):
        plot_curve(axes[1, 1], runs[s['name']], f'z={strength}', '-')
    plot_curve(axes[1, 1], runs[BASELINE['name']], 'Baseline', 'g--')
    axes[1, 1].set_xlabel('Round')
    axes[1, 1].set_ylabel('Validation Accuracy')
    axes[1, 1].set_title('Training Progress Over Rounds (validation split)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(f'Naive Reversed/Scaled Poisoning Attack: mean ± std over valid seeds '
                 f'(of {baseline["num_seeds"]}; diverged/degenerate runs excluded)', fontsize=14)
    plt.tight_layout()
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, 'experiment_results.png'), dpi=150, bbox_inches='tight')
    plt.close()

def print_summary(stats, results_dir, plots_dir):
    baseline = stats[BASELINE['name']]
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY (test split, mean ± std over valid seeds)")
    print("Diverged (non-finite outputs) and degenerate (single predicted class) runs are")
    print("excluded from every mean; [v/n valid] gives the seeds behind each mean.")
    print("=" * 70)
    print(f"Configuration: {NUM_CLIENTS} clients, {NUM_MALICIOUS} malicious, seeds {baseline['seeds']}")
    print(f"Baseline (No Attack): acc {format_metric(baseline['test_accuracy'], baseline)}, "
          f"balanced {format_metric(baseline['test_balanced_accuracy'], baseline)}")
    print("\nAttack Strength Results (FedAvg):")
    for spec, z in zip(ATTACK_SWEEP, ATTACK_STRENGTHS):
        s = stats[spec['name']]
        print(f"  z={z}: acc {format_metric(s['test_accuracy'], s)}, "
              f"balanced {format_metric(s['test_balanced_accuracy'], s)}, "
              f"magnitude {format_metric(s['stealthiness'], s, digits=2)}")
    print(f"\nDefense Results (no attack vs. attack z={DEFENSE_ATTACK_STRENGTH}):")
    for control, attacked in DEFENSE_PAIRS:
        print(f"  {DEFENSE_LABELS[attacked['defense']['name']]}:")
        for tag, spec in [('no attack', control), ('attack', attacked)]:
            s = stats[spec['name']]
            line = (f"    {tag:<10} acc {format_metric(s['test_accuracy'], s)}, "
                    f"balanced {format_metric(s['test_balanced_accuracy'], s)}")
            if s['detection_tpr'] is not None:
                line += f", detection TPR {format_metric(s['detection_tpr'], s)}"
            if s['detection_fpr'] is not None:
                line += f", FPR {format_metric(s['detection_fpr'], s)}"
            print(line)
    print("\nPer-seed results (FedAvg; attacker share = attackers' FedAvg weight):")
    print(f"  {'config':<30} {'seed':>4}  {'status':<10} {'1st non-finite':>14}  "
          f"{'attacker share':>14}  {'acc':>6}  {'balanced':>8}")
    for spec in [BASELINE] + ATTACK_SWEEP:
        for row in stats[spec['name']]['per_seed']:
            first = row['first_nonfinite_round']
            share = row['attacker_data_share']
            print(f"  {spec['name']:<30} {row['seed']:>4}  {row['status']:<10} "
                  f"{'round ' + str(first) if first else '—':>14}  "
                  f"{'—' if share is None else f'{share:.1%}':>14}  "
                  f"{row['test_accuracy']:>6.4f}  {row['test_balanced_accuracy']:>8.4f}")
    excluded = [(name, row) for name, s in stats.items() for row in s['per_seed']
                if row['status'] != 'valid']
    if excluded:
        print("\nExcluded runs:")
        for name, row in excluded:
            first = row['first_nonfinite_round']
            when = f", first non-finite validation output in round {first}" if first else ''
            print(f"  {name}, seed {row['seed']}: {row['status']}{when}")
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
