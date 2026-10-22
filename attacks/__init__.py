from .reversed_scaled_attack import reversed_scaled_attack

# Update-manipulation attacks: fn(global_weights, benign_weights, **params) -> poisoned weights
ATTACKS = {
    'reversed_scaled': reversed_scaled_attack,
}

def apply_attack(name, global_weights, benign_weights, **params):
    if name not in ATTACKS:
        raise ValueError(f"Unknown attack '{name}' (known: {sorted(ATTACKS)})")
    return ATTACKS[name](global_weights, benign_weights, **params)

__all__ = ['reversed_scaled_attack', 'ATTACKS', 'apply_attack']
