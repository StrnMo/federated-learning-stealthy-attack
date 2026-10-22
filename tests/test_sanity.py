"""
Sanity checks for the FL pipeline. Run from the repo root with:

    python -m pytest tests -q
"""

import os
import sys
import json
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'experiments'))

from attacks import apply_attack
from attacks.reversed_scaled_attack import reversed_scaled_attack
from defenses.aggregation import (fedavg_aggregate, trimmed_mean_aggregate, krum_aggregate, krum_scores,
                                  multi_krum_select, multi_krum_aggregate, aggregate)
from models.cnn_model import get_model
from utils.data_utils import FLClient
from utils.metrics import classification_metrics, confusion_matrix, detection_rates
import run_experiment as rx

def weights(*values):
    return {'w': torch.tensor(values, dtype=torch.float32)}

# ---------- Unit checks on synthetic tensors ----------

def test_attack_reverses_and_scales_update():
    poisoned = reversed_scaled_attack(weights(1.0, 2.0), weights(1.5, 1.0), scale=3.0)
    assert torch.allclose(poisoned['w'], torch.tensor([1.0 - 3 * 0.5, 2.0 - 3 * -1.0]))

def test_fedavg_without_counts_is_mean():
    agg = fedavg_aggregate([weights(0.0), weights(2.0), weights(4.0)])
    assert torch.allclose(agg['w'], torch.tensor([2.0]))

def test_fedavg_weights_by_sample_count():
    agg = fedavg_aggregate([weights(0.0, 1.0), weights(4.0, 5.0)], num_samples=[300, 100])
    assert torch.allclose(agg['w'], torch.tensor([1.0, 2.0]))  # 0.75*w1 + 0.25*w2

def test_fedavg_rejects_mismatched_counts():
    with pytest.raises(ValueError):
        fedavg_aggregate([weights(0.0), weights(1.0)], num_samples=[10])

def test_trimmed_mean_drops_extremes():
    clients = [weights(float(v)) for v in [-100, 1, 2, 3, 4, 5, 6, 7, 8, 100]]
    agg = trimmed_mean_aggregate(clients, trim_ratio=0.2)  # trims 2 from each end
    assert torch.allclose(agg['w'], torch.tensor([4.5]))

def test_krum_rejects_outliers():
    clients = [weights(float(v)) for v in [0.0, 0.1, -0.1, 0.2, -0.2, 0.05, -0.05, 0.15]]
    clients += [weights(100.0), weights(100.0)]
    agg = krum_aggregate(clients, num_malicious=2)
    assert abs(agg['w'].item()) < 1.0

# n=6, f=1: plain-distance Krum would pick client 2, squared-distance Krum picks client 0
KRUM_POINTS = [(-5, 0), (-5, -5), (-5, -4), (2, -1), (2, 2), (-4, 5)]

def test_krum_uses_squared_distances():
    clients = [weights(float(x), float(y)) for x, y in KRUM_POINTS]
    assert krum_scores(clients, num_malicious=1).tolist() == [67, 91, 75, 117, 107, 143]
    assert torch.equal(krum_aggregate(clients, num_malicious=1)['w'], torch.tensor([-5.0, 0.0]))

def test_krum_rejects_too_many_attackers():
    # n must exceed 2f + 2; previously this silently fell back to trimmed mean
    with pytest.raises(ValueError):
        krum_aggregate([weights(float(v)) for v in range(6)], num_malicious=2)

def test_model_state_is_only_trainable_parameters():
    # No normalisation buffers that could be averaged across clients or flipped by an attack
    model = get_model(8, 3)
    assert list(model.buffers()) == []
    assert set(model.state_dict()) == {name for name, _ in model.named_parameters()}

def test_model_output_stays_finite_under_reversed_update():
    # With BatchNorm, reversing running_var made it negative and every logit NaN
    model = get_model(8, 3)
    g = {k: v.clone() for k, v in model.state_dict().items()}
    w = {k: v + 0.5 * torch.randn_like(v) for k, v in g.items()}
    model.load_state_dict(reversed_scaled_attack(g, w, scale=5.0))
    model.eval()
    assert torch.isfinite(model(torch.rand(16, 3, 28, 28))).all()

TINY_DATA = TensorDataset(torch.rand(64, 3, 28, 28), torch.randint(0, 8, (64, 1)))

def tiny_client(**kwargs):
    return FLClient(0, TINY_DATA, get_model(8, 3), torch.device('cpu'), batch_size=32,
                    local_epochs=1, seed=0, **kwargs)

def test_client_uses_fresh_sgd_each_round():
    client = tiny_client(lr=0.01, momentum=0.9)
    client.train()
    first = client.optimizer
    client.train()
    assert isinstance(client.optimizer, torch.optim.SGD)
    assert client.optimizer is not first
    group = client.optimizer.param_groups[0]
    assert (group['lr'], group['momentum']) == (0.01, 0.9)

def test_local_update_ignores_previous_rounds():
    # Same start weights and same batches must give the same update, whatever came before
    start = {k: v.clone() for k, v in get_model(8, 3).state_dict().items()}
    fresh, used = tiny_client(), tiny_client()
    used.set_weights(start)
    used.train()  # one extra round of history
    updates = []
    for c in (fresh, used):
        c.dataloader.generator.manual_seed(123)  # same batch order
        torch.manual_seed(123)                   # same dropout masks
        c.set_weights(start)
        updates.append(c.train())
    a, b = updates
    assert all(torch.equal(a[k], b[k]) for k in a)

def test_multi_krum_averages_lowest_scoring_clients():
    clients = [weights(float(x), float(y)) for x, y in KRUM_POINTS]
    # Scores [67, 91, 75, 117, 107, 143]: the 3 lowest are clients 0, 2, 1
    assert multi_krum_select(clients, num_malicious=1, m=3) == [0, 1, 2]
    agg = multi_krum_aggregate(clients, num_malicious=1, m=3)
    assert torch.allclose(agg['w'], torch.tensor([-5.0, -3.0]))  # mean of (-5,0), (-5,-5), (-5,-4)

def test_multi_krum_drops_outliers_with_default_m():
    clients = [weights(float(v)) for v in [0.0, 0.1, -0.1, 0.2, -0.2, 0.05, -0.05, 0.15]]
    clients += [weights(100.0), weights(100.0)]
    agg, kept = aggregate('multi_krum', clients, [10] * 10, 2)  # m = n - f = 8
    assert kept == list(range(8))
    assert abs(agg['w'].item()) < 1.0
    assert detection_rates(kept, range(10), {8, 9}) == {'tpr': 1.0, 'fpr': 0.0}

def test_multi_krum_rejects_invalid_m():
    clients = [weights(float(x), float(y)) for x, y in KRUM_POINTS]
    with pytest.raises(ValueError):
        multi_krum_select(clients, num_malicious=1, m=0)

def test_defense_registry_reports_kept_clients():
    clients = [weights(float(x), float(y)) for x, y in KRUM_POINTS]
    _, kept = aggregate('krum', clients, [10] * 6, 1)
    assert kept == [0]
    _, kept = aggregate('trimmed_mean', clients, [10] * 6, 1, trim_ratio=0.2)
    assert kept is None
    with pytest.raises(ValueError):
        aggregate('no_such_defense', clients, [10] * 6, 1)

def test_attack_registry_matches_direct_call():
    g, w = weights(1.0, 2.0), weights(1.5, 1.0)
    assert torch.equal(apply_attack('reversed_scaled', g, w, scale=3.0)['w'],
                       reversed_scaled_attack(g, w, scale=3.0)['w'])
    with pytest.raises(ValueError):
        apply_attack('no_such_attack', g, w)

def test_confusion_matrix_and_class_metrics():
    y_true = [0, 0, 0, 0, 1, 1, 2, 2]
    y_pred = [0, 0, 0, 1, 1, 0, 2, 2]
    cm = confusion_matrix(y_true, y_pred, num_classes=3)
    assert cm.tolist() == [[3, 1, 0], [1, 1, 0], [0, 0, 2]]
    m = classification_metrics(cm)
    assert m['accuracy'] == 6 / 8
    assert m['per_class_recall'] == [0.75, 0.5, 1.0]
    assert m['balanced_accuracy'] == pytest.approx((0.75 + 0.5 + 1.0) / 3)
    assert 'attack_success_rate' not in m  # untargeted

def test_balanced_accuracy_ignores_absent_classes():
    m = classification_metrics([[2, 0, 0], [0, 0, 0], [1, 0, 1]])
    assert m['per_class_recall'] == [1.0, None, 0.5]
    assert m['balanced_accuracy'] == 0.75

def test_attack_success_rate_excludes_target_class_samples():
    # Target class 1: its own samples (row 1) must not count, whatever they are predicted as
    cm = [[6, 2, 0],
          [0, 5, 0],
          [1, 1, 2]]
    m = classification_metrics(cm, target_class=1)
    assert m['attack_success_rate'] == 3 / 12

def test_detection_rates():
    participants, malicious = list(range(20)), {0, 1, 2, 3}
    assert detection_rates([7], participants, malicious) == {'tpr': 1.0, 'fpr': 15 / 16}
    assert detection_rates([0], participants, malicious) == {'tpr': 0.75, 'fpr': 1.0}
    assert detection_rates([7], participants, set()) == {'tpr': None, 'fpr': 19 / 20}
    assert detection_rates(None, participants, malicious) is None

def test_device_is_gpu_when_available():
    from utils.data_utils import DEVICE
    assert DEVICE.type == ('cuda' if torch.cuda.is_available() else 'cpu')

def test_data_root_follows_environment_variable(tmp_path):
    import subprocess
    code = "from utils.data_utils import DATA_ROOT; print(DATA_ROOT)"
    env = {**os.environ, 'MEDMNIST_ROOT': str(tmp_path)}
    out = subprocess.run([sys.executable, '-c', code], cwd=REPO_ROOT, env=env,
                         capture_output=True, text=True, check=True).stdout.strip().splitlines()
    assert out[-1] == str(tmp_path)

def test_environment_info_records_device():
    from utils.reproducibility import get_environment_info
    info = get_environment_info(torch.device('cpu'))
    assert info['device'] == 'cpu' and info['gpu'] is None
    assert info['torch'] == torch.__version__

# Configurations already run on Colab; renaming them would orphan the saved results
EXISTING_CONFIG_NAMES = {
    'fedavg_no_attack', 'fedavg_reversed_scaled_z1.5', 'fedavg_reversed_scaled_z2.0',
    'fedavg_reversed_scaled_z3.0', 'fedavg_reversed_scaled_z5.0',
    'trimmed_mean_reversed_scaled_z3.0', 'krum_reversed_scaled_z3.0',
}

def test_every_defense_has_a_no_attack_control():
    for control, attacked in rx.DEFENSE_PAIRS:
        assert control['attack'] is None and attacked['attack'] is not None
        assert control['defense'] == attacked['defense']
    names = [s['name'] for s in rx.CONFIGS]
    assert len(names) == len(set(names))
    assert {'trimmed_mean_no_attack', 'krum_no_attack', 'multi_krum_no_attack',
            'multi_krum_reversed_scaled_z3.0'} <= set(names)
    assert EXISTING_CONFIG_NAMES <= set(names)

def fake_run(seed, accuracy=0.8, status='valid', nonfinite_from_round=None, rounds=3,
             client_sizes=None, malicious_ids=(0, 1, 2, 3)):
    """Minimal saved-run dict, as loaded from a result JSON"""
    cm = np.zeros((8, 8), dtype=int)
    if status == 'valid':
        np.fill_diagonal(cm, 10)
        cm[0, 1] = 5
    else:
        cm[:, 0] = 10  # every prediction in class 0 (what argmax of NaN rows also gives)
    test = {**classification_metrics(cm), 'accuracy': accuracy, 'confusion_matrix': cm.tolist(),
            'nonfinite_samples': 3421 if status == 'diverged' else 0}
    history = [{'round': r, 'val_accuracy': accuracy, 'val_balanced_accuracy': accuracy,
                'val_nonfinite_samples': 1712 if nonfinite_from_round and r >= nonfinite_from_round else 0,
                'kept_clients': None, 'detection': None} for r in range(1, rounds + 1)]
    return {'seed': seed, 'test': test, 'history': history,
            'client_sizes': client_sizes or [100] * 20, 'malicious_ids': list(malicious_ids),
            'stealthiness': float('nan') if status == 'diverged' else 5.0}

def test_run_status_flags_diverged_and_degenerate_runs():
    assert rx.run_status(fake_run(0)) == 'valid'
    assert rx.run_status(fake_run(0, status='diverged')) == 'diverged'
    assert rx.run_status(fake_run(0, status='degenerate')) == 'degenerate'
    assert rx.first_nonfinite_round(fake_run(0, status='diverged', nonfinite_from_round=2)) == 2
    assert rx.first_nonfinite_round(fake_run(0)) is None

def test_summary_excludes_diverged_and_degenerate_runs():
    runs = [fake_run(0, 0.8), fake_run(1, 0.6),
            fake_run(2, 0.0713, 'diverged', nonfinite_from_round=2),
            fake_run(3, 0.19, 'degenerate'), fake_run(4, 0.0713, 'diverged')]
    s = rx.summarise(runs)
    assert (s['num_valid'], s['num_diverged'], s['num_degenerate']) == (2, 2, 1)
    assert (s['valid_seeds'], s['diverged_seeds'], s['degenerate_seeds']) == ([0, 1], [2, 4], [3])
    assert s['test_accuracy']['mean'] == pytest.approx(0.7)  # valid seeds only
    assert s['stealthiness']['mean'] == 5.0                  # no NaN from the diverged runs
    row = s['per_seed'][2]
    assert (row['seed'], row['status'], row['first_nonfinite_round']) == (2, 'diverged', 2)
    assert s['per_seed'][4]['first_nonfinite_round'] is None  # non-finite on test only
    assert [r['test_accuracy'] for r in s['per_seed']] == [0.8, 0.6, 0.0713, 0.19, 0.0713]
    assert rx.counts_label(s) == '[2/5 valid: 2 diverged, 1 degenerate]'
    assert rx.format_metric(s['test_accuracy'], s).endswith('[2/5 valid: 2 diverged, 1 degenerate]')

def test_summary_reports_nothing_when_no_run_is_valid():
    s = rx.summarise([fake_run(0, status='diverged'), fake_run(1, status='degenerate')])
    assert s['test_accuracy'] is None and s['stealthiness'] is None
    assert s['test_per_class_recall_mean'] is None
    assert rx.format_metric(s['test_accuracy'], s) == '— [0/2 valid: 1 diverged, 1 degenerate]'
    assert rx.counts_label(rx.summarise([fake_run(0)])) == '[1/1 valid]'

def test_plot_and_summary_handle_fully_excluded_configs(tmp_path, capsys):
    # FedAvg at z >= 2 diverged on every seed in the Colab results
    bad = {s['name'] for s in rx.ATTACK_SWEEP[1:]}
    runs = {spec['name']: [fake_run(seed, status='diverged' if spec['name'] in bad else 'valid')
                           for seed in range(3)] for spec in rx.CONFIGS}
    stats = {name: rx.summarise(r) for name, r in runs.items()}
    rx.plot_results(runs, stats, str(tmp_path))
    rx.print_summary(stats, str(tmp_path), str(tmp_path))
    height, width = plt.imread(tmp_path / 'experiment_results.png').shape[:2]
    assert width < 3000 and height < 2500  # labels stay inside the figure
    out = capsys.readouterr().out
    assert '— [0/3 valid: 3 diverged]' in out and 'seed 2: diverged' in out
    assert 'Per-seed results (FedAvg' in out and '20.0%' in out

def test_attacker_data_share_is_their_fedavg_weight():
    # Clients 0-3 attack; client 19 has too few samples and does not participate
    sizes = [300, 100, 50, 50] + [100] * 15 + [rx.PARTITION['min_samples'] - 1]
    run = fake_run(0, client_sizes=sizes)
    assert rx.attacker_data_share(run) == pytest.approx(500 / 2000)
    assert rx.attacker_data_share(fake_run(0, malicious_ids=())) is None  # no-attack run

# ---------- End-to-end checks on BloodMNIST (short runs) ----------

@pytest.fixture(scope='module')
def short_runs():
    attacked = rx.make_spec('fedavg', 3.0)
    return {
        'a': rx.run_experiment(attacked, seed=0, num_rounds=1),
        'b': rx.run_experiment(attacked, seed=0, num_rounds=1),
        'c': rx.run_experiment(rx.BASELINE, seed=0, num_rounds=1),
        'd': rx.run_experiment(rx.BASELINE, seed=1, num_rounds=1),
        'krum': rx.run_experiment(rx.make_spec('krum', 3.0), seed=0, num_rounds=1),
    }

def test_same_seed_and_config_is_identical(short_runs):
    assert short_runs['a'] == short_runs['b']

def test_same_seed_gives_same_partition_across_configs(short_runs):
    assert short_runs['a']['client_sizes'] == short_runs['c']['client_sizes']

def test_different_seed_gives_different_partition(short_runs):
    assert short_runs['c']['client_sizes'] != short_runs['d']['client_sizes']

def test_baseline_has_no_attackers(short_runs):
    assert short_runs['c']['malicious_ids'] == []
    assert short_runs['a']['malicious_ids'] == [0, 1, 2, 3]

def test_run_records_which_clients_each_defense_kept(short_runs):
    (fedavg_round,) = short_runs['a']['history']
    assert fedavg_round['kept_clients'] is None and fedavg_round['detection'] is None
    (krum_round,) = short_runs['krum']['history']
    kept = krum_round['kept_clients']
    assert len(kept) == 1 and 0 <= kept[0] < rx.NUM_CLIENTS
    expected = detection_rates(kept, range(rx.NUM_CLIENTS), short_runs['krum']['malicious_ids'])
    assert krum_round['detection'] == expected

def test_run_reports_final_test_metrics(short_runs):
    test = short_runs['a']['test']
    cm = test['confusion_matrix']
    assert len(cm) == 8 and sum(map(sum, cm)) == 3421  # whole BloodMNIST test split
    assert test['nonfinite_samples'] == 0
    assert test['balanced_accuracy'] == pytest.approx(classification_metrics(cm)['balanced_accuracy'])

def test_run_wiring_devices_counts_and_splits(monkeypatch):
    # Client updates live on CPU, so a CUDA global state would crash the attack and aggregation.
    # Only catches a regression when CUDA is available; on CPU-only machines it passes trivially.
    devices = set()
    def record(*dicts):
        for d in dicts:
            devices.update(v.device.type for v in d.values())
    
    real_attack, real_aggregate = rx.apply_attack, rx.aggregate
    def attack(name, g, w, **params):
        record(g, w)
        return real_attack(name, g, w, **params)
    reported_counts = []
    def aggregate(name, ws, ns, f, **params):
        record(*ws)
        reported_counts.append(ns)
        return real_aggregate(name, ws, ns, f, **params)
    evaluated_splits = []
    real_evaluate = rx.evaluate
    def evaluate(model, loader, *args):
        evaluated_splits.append(loader.dataset.split)
        return real_evaluate(model, loader, *args)
    monkeypatch.setattr(rx, 'apply_attack', attack)
    monkeypatch.setattr(rx, 'aggregate', aggregate)
    monkeypatch.setattr(rx, 'evaluate', evaluate)
    
    result = rx.run_experiment(rx.make_spec('fedavg', 3.0), seed=0, num_rounds=2)
    assert devices == {'cpu'}
    # Validation every round, test exactly once at the end
    assert evaluated_splits == ['val', 'val', 'test']
    # The aggregator receives every active client's true sample count
    active_sizes = [n for n in result['client_sizes'] if n >= rx.PARTITION['min_samples']]
    assert reported_counts == [active_sizes, active_sizes]

def test_pipeline_writes_results_and_resumes(tmp_path, monkeypatch):
    results, plots = tmp_path / 'results', tmp_path / 'plots'
    def main(*args):
        monkeypatch.setattr(sys, 'argv', ['run_experiment.py', '--results-dir', str(results),
                                          '--plots-dir', str(plots), *args])
        rx.main()

    # Fresh run writes every (seed, spec) file, the summary and the plot
    main('--seeds', '0', '1', '--rounds', '1')
    expected = {f"seed{s}_{spec['name']}.json" for s in (0, 1) for spec in rx.CONFIGS}
    assert set(os.listdir(results)) == expected | {'summary.json'}
    assert (plots / 'experiment_results.png').exists()
    summary = json.loads((results / 'summary.json').read_text())
    assert summary['summary'][rx.BASELINE['name']]['num_seeds'] == 2
    run = json.loads((results / f"seed0_{rx.BASELINE['name']}.json").read_text())
    assert run['spec'] == rx.BASELINE and run['config']['partition'] == rx.PARTITION

    # --summary-only trains nothing and rebuilds the same summary from the saved runs
    real_run = rx.run_experiment
    trained = []
    def counting_run(spec, seed, num_rounds):
        trained.append((seed, spec['name']))
        return real_run(spec, seed, num_rounds)
    monkeypatch.setattr(rx, 'run_experiment', counting_run)
    (results / 'summary.json').unlink()
    main('--seeds', '0', '1', '--rounds', '1', '--summary-only')
    assert trained == []
    assert json.loads((results / 'summary.json').read_text())['summary'] == summary['summary']

    # Resume: only the missing run is trained; --skip-summary writes no summary
    missing = rx.CONFIGS[-1]
    (results / f"seed1_{missing['name']}.json").unlink()
    (results / 'summary.json').unlink()
    main('--seeds', '1', '--rounds', '1', '--skip-summary')
    assert trained == [(1, missing['name'])]
    assert (results / f"seed1_{missing['name']}.json").exists()
    assert not (results / 'summary.json').exists()
    assert not any(name.endswith('.tmp') for name in os.listdir(results))

    # Missing seeds and runs with different settings are errors, never silently mixed
    with pytest.raises(FileNotFoundError):
        main('--seeds', '0', '1', '2', '--rounds', '1', '--summary-only')
    with pytest.raises(ValueError, match='num_rounds'):
        main('--seeds', '0', '--rounds', '2', '--summary-only')
