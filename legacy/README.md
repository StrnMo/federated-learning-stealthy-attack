# legacy/

This folder is kept for the record only. Do not use it for new results.

## run_stealthy_analysis.py

Restored unchanged from commit `8ec72cd` (path `experiments/run_stealthy_analysis.py`; blob `e1ab3b5`). It was deleted in commit `b390b68`.

- **It produced the numbers in the `8ec72cd` README,** for example a 91.96% baseline and 90.32% Trimmed Mean / 58.70% Krum at z = 3. It did **not** produce the numbers in the current README (90.0% baseline, 84.5% Trimmed Mean, 68.9% Krum). No committed code produced those.
- **It uses boosted updates,** `w_g + z·(w_local − w_g)`, not the sign-flipped `w_g − z·(w_local − w_g)` used by the current `attacks/baruch_attack.py`.
- **It trains for 3 local epochs,** not 2.
- **It was run without seeds,** so its results cannot be reproduced exactly.
- **It no longer runs as-is.** It imports `defenses.robust_aggregation`, which was removed in `d66f937`. That module can be viewed with `git show 8ec72cd:defenses/robust_aggregation.py`.
