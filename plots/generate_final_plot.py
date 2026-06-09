import matplotlib.pyplot as plt
import numpy as np

# Results from your experiment
defenses = ['FedAvg\n(No Defense)', 'Trimmed Mean', 'Krum']
accuracies = [0.0713, 0.78, 0.75]  # Update with your actual Krum result

plt.figure(figsize=(10, 6))
bars = plt.bar(defenses, accuracies, color=['red', 'green', 'blue'])
plt.axhline(y=0.90, color='black', linestyle='--', label='Baseline (No Attack): 90%')
plt.ylabel('Test Accuracy')
plt.xlabel('Defense Method')
plt.title('Defense Effectiveness Against Model Poisoning Attack (z=3.0)')
plt.ylim(0, 1)

for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
             f'{acc*100:.1f}%', ha='center', fontsize=12)

plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('plots/final_defense_results.png', dpi=150)
plt.show()
print("✅ Plot saved to plots/final_defense_results.png")


