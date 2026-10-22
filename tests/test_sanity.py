"""
Sanity checks for the FL pipeline. Run from the repo root with:

    python -m pytest tests -q
"""

import os
import sys
import json
import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'experiments'))

from attacks.baruch_attack import baruch_attack
from defenses.aggregation import fedavg_aggregate, trimmed_mean_aggregate, krum_aggregate
import run_experiment as rx

def weights(*values):
    return {'w': torch.tensor(values, dtype=torch.float32)}

# ---------- Unit checks on synthetic tensors ----------

def test_attack_reverses_and_scales_update():
    poisoned = baruch_attack(weights(1.0, 2.0), weights(1.5, 1.0), attack_strength=3.0)
    assert torch.allclose(poisoned['w'], torch.tensor([1.0 - 3 * 0.5, 2.0 - 3 * -1.0]))

def test_fedavg_is_mean():
    agg = fedavg_aggregate([weights(0.0), weights(2.0), weights(4.0)])
    assert torch.allclose(agg['w'], torch.tensor([2.0]))

def test_trimmed_mean_drops_extremes():
    clients = [weights(float(v)) for v in [-100, 1, 2, 3, 4, 5, 6, 7, 8, 100]]
    agg = trimmed_mean_aggregate(clients, trim_ratio=0.2)  # trims 2 from each end
    assert torch.allclose(agg['w'], torch.tensor([4.5]))

def test_krum_rejects_outliers():
    clients = [weights(float(v)) for v in [0.0, 0.1, -0.1, 0.2, -0.2, 0.05, -0.05, 0.15]]
    clients += [weights(100.0), weights(100.0)]
    agg = krum_aggregate(clients, num_malicious=2)
    assert abs(agg['w'].item()) < 1.0

# ---------- End-to-end checks on BloodMNIST (short runs) ----------

@pytest.fixture(scope='module')
def short_runs():
    return {
        'a': rx.run_experiment(3.0, 'fedavg', seed=0, num_rounds=1),
        'b': rx.run_experiment(3.0, 'fedavg', seed=0, num_rounds=1),
        'c': rx.run_experiment(1.0, 'fedavg', seed=0, num_rounds=1),
        'd': rx.run_experiment(1.0, 'fedavg', seed=1, num_rounds=1),
    }

def test_same_seed_and_config_is_identical(short_runs):
    assert short_runs['a'] == short_runs['b']

def test_same_seed_gives_same_partition_across_configs(short_runs):
    assert short_runs['a']['client_sizes'] == short_runs['c']['client_sizes']

def test_different_seed_gives_different_partition(short_runs):
    assert short_runs['c']['client_sizes'] != short_runs['d']['client_sizes']

def test_pipeline_writes_results(tmp_path, monkeypatch):
    monkeypatch.setattr(rx, 'RESULTS_DIR', str(tmp_path / 'results'))
    monkeypatch.setattr(rx, 'PLOTS_DIR', str(tmp_path / 'plots'))
    monkeypatch.setattr(sys, 'argv', ['run_experiment.py', '--seeds', '0', '1', '--rounds', '1'])
    rx.main()

    files = set(os.listdir(tmp_path / 'results'))
    expected = {f'seed{s}_{rx.config_name(*cfg)}.json' for s in (0, 1) for cfg in rx.CONFIGS}
    assert expected | {'summary.json'} == files
    assert (tmp_path / 'plots' / 'experiment_results.png').exists()

    summary = json.loads((tmp_path / 'results' / 'summary.json').read_text())
    assert summary['summary'][rx.config_name(*rx.BASELINE)]['num_seeds'] == 2
