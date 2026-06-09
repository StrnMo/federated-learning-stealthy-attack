"""
Main experiment script for stealthy model poisoning attacks
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import copy
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Subset

from models.cnn_model import get_model
from attacks.baruch_attack import baruch_attack
from defenses.aggregation import fedavg_aggregate, trimmed_mean_aggregate, krum_aggregate
from utils.data_utils import load_data, create_non_iid_splits, FLClient, DEVICE
from utils.metrics import compute_stealthiness, evaluate_model

# Experiment Configuration
NUM_CLIENTS = 20
NUM_ROUNDS = 30
NUM_MALICIOUS = 4
LOCAL_EPOCHS = 2
BATCH_SIZE = 32
MIN_SAMPLES_PER_CLIENT = 30
ATTACK_STRENGTHS = [1.5, 2.0, 3.0, 5.0]

os.makedirs('plots', exist_ok=True)
os.makedirs('results', exist_ok=True)

def run_experiment(attack_strength, defense='fedavg'):
    """Run single FL experiment with given attack strength and defense"""
    
    # Load data
    train_data, test_loader, num_classes = load_data()
    client_indices = create_non_iid_splits(train_data, NUM_CLIENTS, alpha=0.5, 
                                           min_samples=MIN_SAMPLES_PER_CLIENT)
    
    # Initialize clients
    base_model = get_model(num_classes, input_channels=3).to(DEVICE)
    clients = []
    for i, indices in enumerate(client_indices):
        if len(indices) >= MIN_SAMPLES_PER_CLIENT:
            subset = Subset(train_data, indices)
            clients.append(FLClient(i, subset, copy.deepcopy(base_model), DEVICE, 
                                    BATCH_SIZE, LOCAL_EPOCHS))
        else:
            clients.append(None)
    
    active_clients = [c for c in clients if c is not None and c.has_data]
    num_valid_clients = len(active_clients)
    print(f"  Active clients: {num_valid_clients}")
    
    if num_valid_clients < 10:
        return {'final_accuracy': 0, 'stealthiness': 0}
    
    actual_malicious = min(NUM_MALICIOUS, max(2, num_valid_clients // 5))
    print(f"  Malicious clients: {actual_malicious}")
    
    # Initialize global model
    global_model = copy.deepcopy(base_model)
    global_weights = global_model.state_dict()
    
    accuracies = []
    stealthiness_scores = []
    
    # Federated training rounds
    for round_idx in range(NUM_ROUNDS):
        client_weights = []
        
        for i, client in enumerate(clients):
            if client is None or not client.has_data:
                continue
            
            # Local training
            client.set_weights(copy.deepcopy(global_weights))
            honest_weights = client.train()
            
            if honest_weights is None:
                continue
            
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
            global_weights = fedavg_aggregate(client_weights)
        
        if global_weights is not None:
            global_model.load_state_dict(global_weights)
            acc = evaluate_model(global_model, test_loader, DEVICE)
            accuracies.append(acc)
            
            if (round_idx + 1) % 10 == 0:
                print(f"    Round {round_idx + 1}: Acc = {acc:.4f}")
    
    return {
        'final_accuracy': accuracies[-1] if accuracies else 0,
        'stealthiness': np.mean(stealthiness_scores) if stealthiness_scores else 0,
        'accuracies': accuracies
    }

def plot_results(baseline, results, defense_results):
    """Generate comprehensive results plots"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    strengths = [r['strength'] for r in results]
    accuracies = [r['accuracy'] for r in results]
    
    # Plot 1: Accuracy vs Attack Strength
    axes[0, 0].plot(strengths, accuracies, 'ro-', linewidth=2, markersize=8, label='FedAvg')
    axes[0, 0].axhline(y=baseline['final_accuracy'], color='g', linestyle='--', 
                       label=f'Baseline: {baseline["final_accuracy"]:.3f}')
    axes[0, 0].set_xlabel('Attack Strength (z)')
    axes[0, 0].set_ylabel('Final Test Accuracy')
    axes[0, 0].set_title('Model Accuracy vs Attack Strength')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Stealthiness (Lower = More Stealthy)
    stealthiness = [r['stealthiness'] for r in results]
    axes[0, 1].plot(strengths, stealthiness, 'bs-', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('Attack Strength (z)')
    axes[0, 1].set_ylabel('Attack Magnitude (L2 norm)')
    axes[0, 1].set_title('Stealthiness (Lower = Harder to Detect)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Defense Comparison
    defense_names = ['FedAvg (No Defense)', 'Trimmed Mean', 'Krum']
    defense_accs = [
        results[2]['accuracy'],  # z=3.0 with FedAvg
        defense_results['trimmed']['final_accuracy'],
        defense_results['krum']['final_accuracy']
    ]
    bars = axes[1, 0].bar(defense_names, defense_accs, color=['red', 'blue', 'green'])
    axes[1, 0].axhline(y=baseline['final_accuracy'], color='black', linestyle='--',
                       label=f'Baseline: {baseline["final_accuracy"]:.3f}')
    axes[1, 0].set_ylabel('Test Accuracy')
    axes[1, 0].set_title('Defense Effectiveness (Attack Strength = 3.0)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, defense_accs):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{val:.3f}', ha='center', fontsize=9)
    
    # Plot 4: Training Progress
    for i, strength in enumerate(strengths[:3]):
        result = results[i]
        axes[1, 1].plot(result['accuracies'], label=f'z={strength}', linewidth=2)
    axes[1, 1].plot(baseline['accuracies'], 'g--', label='Baseline', linewidth=2)
    axes[1, 1].set_xlabel('Round')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_title('Training Progress Over Rounds')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Stealthy Model Poisoning Attack Analysis (Baruch et al. 2019)', fontsize=14)
    plt.tight_layout()
    plt.savefig('plots/experiment_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Print summary
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    print(f"Configuration: {NUM_CLIENTS} clients, {NUM_MALICIOUS} malicious, {NUM_ROUNDS} rounds")
    print(f"Baseline Accuracy (No Attack): {baseline['final_accuracy']:.4f}")
    print("\nAttack Strength Results:")
    print("-" * 50)
    print(f"{'Strength':<10} {'Accuracy':<12} {'Attack Magnitude':<20}")
    print("-" * 50)
    for r in results:
        print(f"z={r['strength']:<5} {r['accuracy']:<12.4f} {r['stealthiness']:<20.2f}")
    print("\nDefense Results (Attack Strength = 3.0):")
    print(f"  FedAvg (No Defense):  {defense_results['fedavg']['final_accuracy']:.4f}")
    print(f"  Trimmed Mean:         {defense_results['trimmed']['final_accuracy']:.4f}")
    print(f"  Krum:                 {defense_results['krum']['final_accuracy']:.4f}")
    print("\n✅ Results saved to plots/experiment_results.png")

def main():
    print("=" * 70)
    print("STEALTHY MODEL POISONING ATTACK EXPERIMENT")
    print(f"Clients: {NUM_CLIENTS}, Malicious: {NUM_MALICIOUS}, Rounds: {NUM_ROUNDS}")
    print("=" * 70)
    
    # Baseline (no attack)
    print("\n[1/4] Running Baseline (No Attack)...")
    baseline = run_experiment(attack_strength=1.0, defense='fedavg')
    
    # Attack strength experiments
    print("\n[2/4] Running Attack Experiments...")
    results = []
    for strength in ATTACK_STRENGTHS:
        print(f"\n  Attack Strength = {strength}")
        result = run_experiment(attack_strength=strength, defense='fedavg')
        results.append({
            'strength': strength,
            'accuracy': result['final_accuracy'],
            'stealthiness': result['stealthiness'],
            'accuracies': result['accuracies']
        })
    
    # Defense comparison (at fixed attack strength)
    print("\n[3/4] Running Defense Comparison (Attack Strength = 3.0)...")
    defense_results = {}
    print("  FedAvg (No Defense)...")
    defense_results['fedavg'] = run_experiment(attack_strength=3.0, defense='fedavg')
    print("  Trimmed Mean Defense...")
    defense_results['trimmed'] = run_experiment(attack_strength=3.0, defense='trimmed_mean')
    print("  Krum Defense...")
    defense_results['krum'] = run_experiment(attack_strength=3.0, defense='krum')
    
    # Generate plots
    print("\n[4/4] Generating Results Plots...")
    plot_results(baseline, results, defense_results)
    
    print("\n✅ Experiment completed successfully!")

if __name__ == "__main__":
    main()