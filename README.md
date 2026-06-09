# Stealthy Model Poisoning Attack for Federated Learning

## 🎯 Overview
This project implements **stealthy model poisoning attacks** (Baruch 2019 - "A Little Is Enough") on federated learning, evaluating both attack effectiveness and defense mechanisms. The work addresses the fundamental question:

> *"How much private information can a malicious participant extract while keeping its poisoned update sufficiently small or stealthy to avoid detection?"*

The implementation demonstrates:
- ✅ Complete model degradation with 4 malicious clients (20%)
- ✅ Effective defense mechanisms (Trimmed Mean, Krum)
- ✅ Clear trade-off between attack strength and model performance


## 📊 Dataset

- **BloodMNIST**: 11,959 training samples, 8 blood cell types
- **Image format**: 28×28 RGB (3 channels)
- **Task**: Multi-class classification (8 classes)
- **Data distribution**: Non-IID (Dirichlet α=0.5)

## 📈 Key Results

### Attack Impact (4 Malicious Clients out of 20)

| Attack Strength (z) | Final Accuracy | Attack Magnitude | Severity |
|--------------------|----------------|------------------|----------|
| No Attack (Baseline) | **90.0%** | - | Healthy |
| z=1.5 (Stealthy) | **84.7%** | Moderate | Mild degradation |
| z=2.0 | **7.1%** | Large | Complete collapse |
| z=3.0 | **7.1%** | Large | Complete collapse |
| z=5.0 | **7.1%** | Very Large | Complete collapse |

**Key Finding:** With only 4 malicious clients (20% participation), attack strength z≥2.0 completely destroys model performance (90% → 7.1%), which is worse than random guessing (12.5% for 8 classes).

### Defense Effectiveness (Attack Strength z=3.0)

| Defense Method | Test Accuracy | Baseline Recovery |
|---------------|--------------|-------------------|
| **FedAvg (No Defense)** | **7.1%** | ❌ Attack succeeds |
| **Trimmed Mean** | **84.5%** | ✅ 94% recovery |
| **Krum** | **68.9%** | ✅ 77% recovery |

**Key Finding:** Trimmed Mean successfully defends against the attack, maintaining 84.5% accuracy (only 5.5% drop from baseline). Krum also provides substantial protection (68.9% accuracy).

### Training Progress

The defense mechanisms show clear superiority over unprotected FedAvg:

```text
Round 10: FedAvg (7%) vs Trimmed Mean (79%) vs Krum (66%)
Round 20: FedAvg (7%) vs Trimmed Mean (79%) vs Krum (61%)
Round 30: FedAvg (7%) vs Trimmed Mean (85%) vs Krum (69%)
```


## Repository Structure

federated-learning-stealthy-attack/
│
├── attacks/
│ ├── __init__.py
│ └── baruch_attack.py # Baruch 2019 model poisoning
│
├── defenses/
│ ├── __init__.py
│ └── aggregation.py # FedAvg, Trimmed Mean, Krum
│
├── models/
│ ├── __init__.py
│ └── cnn_model.py # CNN for BloodMNIST (3×28×28 → 8 classes)
│
├── utils/
│ ├── __init__.py
│ ├── data_utils.py # Data loading, non-IID splits
│ └── metrics.py # Stealthiness, accuracy evaluation
│
├── experiments/
│ ├── run_experiment.py # Main experiment script
│ └── complete_defenses.py # Defense comparison runner
│
├── plots/
│ └── final_defense_results.png
│
├── requirements.txt
└── README.md


## 🚀 Quick Start

### Installation

```bash
# Navigate to project directory
cd federated-learning-stealthy-attack

# Install dependencies
pip install -r requirements.txt
```
### Run Experiments 

```bash
# Full experiment (baseline + attacks + defenses)
python experiments/run_experiment.py

# Or just defense comparison (faster)
python experiments/complete_defenses.py
```


## Technical Details


### Attack Implementation (Baruch 2019)

```python
def baruch_attack(global_weights, benign_weights, attack_strength=2.0):
    """
    Model poisoning attack that moves model OPPOSITE to benign direction.
    Higher attack_strength = more damage but less stealthy.
    """
    poisoned = copy.deepcopy(global_weights)
    for key in poisoned.keys():
        diff = benign_weights[key] - global_weights[key]
        poisoned[key] = global_weights[key] - attack_strength * diff  # ← MINUS sign
    return poisoned
```
### Robust Aggregation Defenses

#### Trimmed Mean: Removes extreme values (largest/smallest 20%) per layer

```python
# Keeps only 60% of client updates (20% trimmed from each end)
trimmed = sorted_values[trim_count:-trim_count]
avg_weights[key] = trimmed.mean(dim=0)
```
#### Krum: Selects the single most consistent update

```python
# Chooses update with smallest sum of distances to neighbors
best_idx = argmin([sum(distances[i, neighbors]) for i in range(num_clients)])
```

### Stealthiness Metric
The L2 norm of the poisoned update. Lower values indicate more stealthy attacks.

```python
def compute_stealthiness(global_weights, poisoned_weights):
    """L2 norm of attack vector - lower = more stealthy"""
    total = 0.0
    for key in global_weights.keys():
        diff = poisoned_weights[key] - global_weights[key]
        total += torch.norm(diff).item() ** 2
    return total ** 0.5
```

### Key Insights 

1. Attack Vulnerability: With only 20% malicious participants, model accuracy collapses from 90% to 7.1%, demonstrating severe vulnerability in standard FedAvg.

2. Defense Effectiveness: Trimmed Mean recovers 94% of baseline performance, showing that robust aggregation can effectively mitigate poisoning attacks.

3. Stealth Trade-off: Lower attack strengths (z=1.5) cause only mild degradation (84.7% accuracy), suggesting potential for stealthy attacks that evade detection.

4. Reproducibility: All experiments use fixed random seeds and clear configuration parameters, ensuring reproducible results

### 🔮 Future Directions
To further explore privacy in federated learning, future work includes:

1. Gradient inversion attacks - Reconstruct individual training samples from model updates and measure PSNR

2. Property inference - Infer whether specific data properties exist in other clients' datasets

3. Membership inference - Determine if a given sample was used in training

4. Privacy-utility trade-off - Quantify leakage vs. model performance degradation

### 📚 References
. Baruch, G., Baruch, M., & Goldberg, Y. (2019). A Little Is Enough: Circumventing Defenses for Distributed Learning. NeurIPS.

. McMahan, B., et al. (2017). Communication-Efficient Learning of Deep Networks from Decentralized Data. AISTATS.

. Yang, J., et al. (2023). *MedMNIST v2: A Large-Scale Lightweight Benchmark for 2D and 3D Biomedical Image Classification.* NeurIPS.


## Technologies

Python 3.8+

PyTorch 2.5.1

MedMNIST

NumPy, Matplotlib