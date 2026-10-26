# Model Poisoning in Federated Learning for Medical Image Classification

**Summary.** Under a naive reversed-and-scaled attack by 4 of 20 clients, FedAvg diverges on almost
every seed. Multi-Krum fully resists the attack (detection TPR 0.99, FPR 0.002), trimmed mean
prevents the collapse but loses about 7 points of balanced accuracy, and Krum fails under non-IID
data even without attack. The next step is a stealthy, class-targeted attack.

This project studies **model poisoning attacks and robust aggregation defenses in federated
learning (FL)** for medical image classification on **BloodMNIST**. Malicious clients send
manipulated model updates to the server, and the project measures how much this harms the global
model and how well robust aggregation rules resist it.

The current attack is a deliberately simple baseline (see [Attack](#attack)). A stealthy,
class-targeted attack is planned (see [Planned work](#planned-work)).

## Setup

| | |
|---|---|
| Dataset | BloodMNIST (MedMNIST v2): 28×28 RGB blood-cell images, 8 classes; official splits, 11,959 train / 1,712 validation / 3,421 test; pixel values scaled to [0, 1] |
| Clients | 20, all participating in every round |
| Data partition | Non-IID label skew, Dirichlet α = 0.5 over the training split; rounding leaves 72–82 training samples (about 0.6%) unassigned, depending on the seed |
| Malicious clients | 4 of 20 (clients 0–3), attacking in every round |
| Training | 30 rounds; each client trains 2 local epochs per round, batch size 32 (the last incomplete batch of each epoch is skipped), SGD (lr 0.01, momentum 0.9), optimizer state reset every round |
| Model | Small CNN: two 3×3 convolutions (32 and 64 channels), each followed by GroupNorm (8 groups), ReLU and 2×2 max-pooling; fully connected 3136 → 128 → 8 with dropout 0.3 before the last layer |
| Aggregation | Sample-weighted FedAvg (weight n_k / Σn), or one of the defenses below; malicious clients report their true sample counts |
| Seeds | 5 (seeds 0–4) per configuration |
| Evaluation | Validation split every round; test split once, after the final round |

**Seeding.** For a given seed, every configuration uses the same data partition, model
initialisation and client data order, so configurations are compared on identical splits. Runs on
GPU are seeded but not bit-for-bit reproducible (GPU kernels are not deterministic), so repeating a
run can give slightly different numbers.

### Attack

The implemented attack is a **naive reversed-and-scaled update**. Each malicious client trains
honestly on its own data, then sends its update reversed and multiplied by a factor z:

    w_sent = w_global − z · (w_local − w_global)

It uses no knowledge of the honest clients and has no stealth constraint: its update is large and
points away from the honest updates, so it is easy to detect. It is **not stealthy** and it is
**not** the "A Little Is Enough" attack of Baruch et al. (2019). It serves as a baseline.

### Defenses

| Defense | Rule | Setting |
|---|---|---|
| FedAvg | Sample-weighted mean (no defense) | — |
| Trimmed mean | Per coordinate, drop the largest and smallest 20% of values, average the rest (Yin et al., 2018) | trim ratio 0.2 |
| Krum | Keep the single client with the lowest Krum score: sum of squared distances to its n − f − 2 nearest neighbours (Blanchard et al., 2017) | f = 4 |
| Multi-Krum | Keep the m clients with the lowest Krum scores and average them (unweighted) | f = 4, m = n − f = 16 |

The defenses are given the true number of malicious clients (f = 4).

### Metrics

- **Main metric: balanced accuracy over all seeds.** Test balanced accuracy (mean recall over the 8
  classes), averaged over all 5 seeds. Runs that broke count at chance level (1/8 = 0.125):
  - **diverged:** the final model produced non-finite outputs (NaN/inf) on the test set;
  - **degenerate:** the outputs are finite, but every test image is assigned the same class.
- **[v/n valid]** next to each value: how many of the n seeds produced a valid (neither diverged
  nor degenerate) run, and why the others are excluded.
- **Detection TPR/FPR** for defenses that keep or remove whole clients (Krum, Multi-Krum): a
  client the defense did not keep counts as flagged. TPR = share of malicious clients flagged,
  FPR = share of honest clients flagged; averaged over rounds, then over the valid seeds.

Means over valid seeds only, per-class recall, confusion matrices and per-seed results are also
recorded by the experiment script.

## Results

Test-set results, mean ± std over seeds. Main metric: balanced accuracy over all 5 seeds, with
diverged and degenerate runs counted as chance (0.125). Chance level is 0.125.

Runs were made on 2026-10-23 on Google Colab (Tesla T4 GPU). The 35 original runs (FedAvg at all
attack strengths, and trimmed mean and Krum under attack) come from commit `58fd005`; the 20 later
runs (the no-attack runs of trimmed mean, Krum and Multi-Krum, and Multi-Krum under attack) come
from commit `d63c71d`. The training code is identical in both commits.

**FedAvg (no defense) at increasing attack strength z**

| Attack | Balanced accuracy (all seeds) |
|---|---|
| No attack | 0.8440 ± 0.0250 [5/5 valid] |
| z = 1.5 | 0.3015 ± 0.2329 [4/5 valid: 1 degenerate] |
| z = 2.0 | 0.2236 ± 0.2206 [1/5 valid: 4 diverged] |
| z = 3.0 | 0.1250 ± 0.0000 [0/5 valid: 4 diverged, 1 degenerate] |
| z = 5.0 | 0.1250 ± 0.0000 [0/5 valid: 5 diverged] |

**Defenses without attack and under attack (z = 3.0)**

| Defense | No attack | Attack z = 3.0 |
|---|---|---|
| FedAvg | 0.8440 ± 0.0250 [5/5 valid] | 0.1250 ± 0.0000 [0/5 valid: 4 diverged, 1 degenerate] |
| Trimmed mean | 0.8126 ± 0.0211 [5/5 valid] | 0.7466 ± 0.0336 [5/5 valid] |
| Krum | 0.3055 ± 0.1170 [4/5 valid: 1 degenerate] | 0.2560 ± 0.1257 [3/5 valid: 2 degenerate] |
| Multi-Krum | 0.8222 ± 0.0302 [5/5 valid] | 0.8247 ± 0.0331 [5/5 valid] |

**Detection by the client-selecting defenses**

| Defense | TPR, attack z = 3.0 | FPR, attack z = 3.0 | FPR, no attack |
|---|---|---|---|
| Multi-Krum | 0.9917 ± 0.0118 [5/5 valid] | 0.0021 ± 0.0029 [5/5 valid] | 0.2000 ± 0.0000 [5/5 valid] |
| Krum | 1.0000 ± 0.0000 [3/5 valid: 2 degenerate] | 0.9375 ± 0.0000 [3/5 valid: 2 degenerate] | 0.9500 ± 0.0000 [4/5 valid: 1 degenerate] |

**How the Krum and Multi-Krum FPR values arise.** Most of these FPR values are fixed by how the
defenses work, not by how well they detect attackers:

- **Krum** keeps exactly 1 of the 20 clients in every round. Without attack, all 19 clients it
  drops are honest, so FPR = 19/20 = 0.95. Under attack, if all 4 attackers are dropped, 15 of the
  16 honest clients are dropped too, so FPR = 15/16 = 0.9375.
- **Multi-Krum** keeps m = 16 clients and drops f = 4 in every round. Without attack, the 4 dropped
  clients are always honest, so FPR = 4/20 = 0.20. Only **under attack** does its FPR reflect
  detection: it dropped the attackers in almost every round (TPR 0.99) and almost never dropped an
  honest client (FPR 0.002).

## Key findings

1. **FedAvg diverges under the attack.** With z ≥ 2, FedAvg breaks on almost every seed: the model
   weights grow until the outputs become non-finite (NaN), after 14–20 rounds at z = 2, 8–10 rounds
   at z = 3 and 7–10 rounds at z = 5. Even at z = 1.5, balanced accuracy drops from 0.84 to 0.30.
2. **Multi-Krum resists the attack; trimmed mean prevents the collapse but still loses about 7
   points** (0.81 without attack, 0.75 under attack).
3. **Krum fails under non-IID data, even without attack** (balanced accuracy 0.31 without attack,
   with 1 of 5 seeds degenerate), most likely because it uses the model of a single client per
   round, and under non-IID data each client holds a skewed subset of the classes.
4. **Robust aggregators cost some accuracy and exclude honest clients.** Without attack, trimmed
   mean costs about 3 points and Multi-Krum about 2 points of balanced accuracy relative to
   FedAvg, and Multi-Krum drops 4 honest clients in every round (FPR 0.20).

## Limitations

- **Naive attack only.** The reversed-and-scaled attack is large and easy to detect; the results
  say nothing about stealthy attacks.
- **One dataset.** Only BloodMNIST has been used.
- **Attackers are fixed to clients 0–3.** Their share of the training data, and so their weight in
  FedAvg, changes from seed to seed (14.8%–25.1% here). The only seed that survived z = 2 under
  FedAvg is the one where the attackers held the least data.
- **The attack-magnitude metric is not yet normalised.** It is the absolute size (L2 norm) of the
  malicious update, not its size relative to the honest updates, so it does not measure stealth.
- **The defenses know the true number of attackers** (f = 4), and the client learning rate has not
  been tuned on the validation split.
- **5 seeds per configuration**, which is too few to rule out effects of the particular partitions.

## Planned work

- A **stealthy, class-targeted poisoning attack** under non-IID data, evaluated against several
  robust aggregation defenses.
- Before that: choose the malicious clients at random (seeded) for each seed, and measure the
  attack magnitude relative to the honest updates. See
  [docs/planned_changes.md](docs/planned_changes.md).

## How to reproduce

### Google Colab (recommended, GPU)

Open [colab/run_experiments.ipynb](colab/run_experiments.ipynb) in Google Colab, select a GPU
runtime and run all cells. The notebook:

1. mounts Google Drive and clones this repository on branch `fix-pipeline`;
2. installs the requirements;
3. runs a short GPU smoke test (1 seed, 2 rounds) and stops if it fails;
4. runs the full experiment one seed at a time, saving every run as a JSON file to a Drive
   folder (finished runs are skipped when the notebook is run again);
5. builds the summary, the results table and the plot from the saved runs.

### Locally

```bash
pip install -r requirements.txt

# Full experiment: 11 configurations × 5 seeds; slow on CPU
python experiments/run_experiment.py --seeds 0 1 2 3 4 --rounds 30 \
    --results-dir results --plots-dir plots

# Rebuild the summary and plot from saved runs without training
python experiments/run_experiment.py --seeds 0 1 2 3 4 --rounds 30 \
    --results-dir results --plots-dir plots --summary-only
```

BloodMNIST is downloaded automatically to `~/.medmnist`, or to the folder given by the
`MEDMNIST_ROOT` environment variable. Each run is saved as `seed{N}_{configuration}.json`;
`summary.json` holds the means, the [v/n valid] counts and the per-seed results.

### Tests

```bash
python -m pytest tests -q
```

The test suite checks the attack, the aggregation rules, the metrics, reproducibility across
seeds and the resumable experiment runner. It takes about 10–15 minutes on CPU.

## Repository structure

```
attacks/reversed_scaled_attack.py   Naive reversed-and-scaled attack
defenses/aggregation.py             FedAvg, trimmed mean, Krum, Multi-Krum
models/cnn_model.py                 GroupNorm CNN for BloodMNIST
utils/data_utils.py                 Data loading, Dirichlet partition, FL client
utils/metrics.py                    Classification and detection metrics
utils/reproducibility.py            Seeding and result files
experiments/run_experiment.py       Experiment runner (all configurations and seeds)
colab/run_experiments.ipynb         Colab notebook for the full run
tests/test_sanity.py                Test suite
docs/planned_changes.md             Changes planned for the next phase
legacy/                             An earlier analysis script, kept for the record only
```

## References

- McMahan, B., Moore, E., Ramage, D., Hampson, S., & Agüera y Arcas, B. (2017).
  Communication-Efficient Learning of Deep Networks from Decentralized Data. *AISTATS*.
- Blanchard, P., El Mhamdi, E. M., Guerraoui, R., & Stainer, J. (2017). Machine Learning with
  Adversaries: Byzantine Tolerant Gradient Descent. *NeurIPS*.
- Yin, D., Chen, Y., Ramchandran, K., & Bartlett, P. (2018). Byzantine-Robust Distributed Learning:
  Towards Optimal Statistical Rates. *ICML*.
- Baruch, G., Baruch, M., & Goldberg, Y. (2019). A Little Is Enough: Circumventing Defenses for
  Distributed Learning. *NeurIPS*.
- Yang, J., Shi, R., Wei, D., Liu, Z., Zhao, L., Ke, B., Pfister, H., & Ni, B. (2023). MedMNIST
  v2 – A large-scale lightweight benchmark for 2D and 3D biomedical image classification.
  *Scientific Data*, 10, 41.
