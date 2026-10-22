"""
Sanity checks for the FL pipeline. Run from the repo root with:

    python -m pytest tests -q
"""

import os
import sys
import json
import pytest
import torch
from torch.utils.data import TensorDataset

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'experiments'))

from attacks.reversed_scaled_attack import reversed_scaled_attack
from defenses.aggregation import fedavg_aggregate, trimmed_mean_aggregate, krum_aggregate, krum_scores
from models.cnn_model import get_model
from utils.data_utils import FLClient
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

def test_server_side_tensors_are_on_cpu(monkeypatch):
    # Client updates live on CPU, so a CUDA global state would crash the attack and aggregation.
    # Only catches a regression when CUDA is available; on CPU-only machines it passes trivially.
    devices = set()
    def record(*dicts):
        for d in dicts:
            devices.update(v.device.type for v in d.values())

    real_attack, real_fedavg = rx.reversed_scaled_attack, rx.fedavg_aggregate
    monkeypatch.setattr(rx, 'reversed_scaled_attack',
                        lambda g, w, z: (record(g, w), real_attack(g, w, z))[1])
    reported_counts = []
    monkeypatch.setattr(rx, 'fedavg_aggregate',
                        lambda ws, ns: (record(*ws), reported_counts.append(ns), real_fedavg(ws, ns))[2])
    result = rx.run_experiment(3.0, 'fedavg', seed=0, num_rounds=1)
    assert devices == {'cpu'}
    # FedAvg receives every active client's true sample count
    active_sizes = [n for n in result['client_sizes'] if n >= rx.MIN_SAMPLES_PER_CLIENT]
    assert reported_counts == [active_sizes]

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
