# Planned changes for the next phase

These are deliberately **not** implemented yet. The current results keep the existing behaviour, so
they stay comparable with the runs already done. All experiments will be rerun for the paper after
these changes.

## 1. Random attacker selection per seed

**Now:** the malicious clients are always clients 0–3 (`malicious_ids` in
`experiments/run_experiment.py`).

**Planned:** choose the malicious clients at random for each seed, using a generator seeded from
that seed, so the choice is reproducible and every configuration with that seed uses the same
attackers.

**Why:** under the Dirichlet split, clients 0–3 hold a different share of the training data in
every seed, and with sample-weighted FedAvg that share is the attackers' total aggregation weight.
Fixing the attacker IDs ties the attack's strength to an arbitrary property of the partition and
probably contributes to the bimodal FedAvg results at z = 1.5 and 2.0. The per-seed table already
reports this share (`attacker_data_share`), so its effect can be checked.

**Affects:** resumed runs. The attacker choice changes the runs, so results produced before and
after this change must not be mixed (use a new results directory).

## 2. Attack magnitude relative to honest-update norms (audit M6)

**Now:** "attack magnitude" (`compute_stealthiness` in `utils/metrics.py`) is the absolute L2 norm
of the malicious update, averaged over rounds. For the reversed/scaled attack this is z times the
attacker's own honest-update norm by construction, so it grows with z by definition and depends on
how training is going (e.g. 4.46 under trimmed mean at z = 3 vs 18.8 under FedAvg at z = 1.5).

**Planned:** measure each malicious update relative to the honest updates of the same round, for
example ‖Δ_malicious‖ / median ‖Δ_honest‖, and report how that compares with the thresholds the
defenses actually use.

**Why:** a stealth metric has to be relative to what a defense compares against. This is needed
for the planned stealthy, class-targeted attack.

## 3. Assign the samples left over by rounding in the Dirichlet partition

**Now:** `create_non_iid_splits` in `utils/data_utils.py` gives each client `int(p_k · n_c)`
samples of class c. The rounding down leaves 72–82 of the 11,959 training samples (about 0.6%)
assigned to no client, depending on the seed.

**Planned:** assign the left-over samples as well, for example by splitting each class at the
rounded cumulative proportions, so that every training sample belongs to exactly one client.

**Why:** all training data should be used; the README currently has to note the unassigned
samples.

**Affects:** every data partition, and therefore all results. Results produced before and after
this change must not be mixed (use a new results directory).

## 4. Per-seed dots in paper figures

**Now:** the summary plot shows mean ± std error bars. When the seeds disagree strongly, this
hides the outcome: for FedAvg at z = 2.0, one seed reaches 0.62 balanced accuracy while the other
four diverge and count at chance (0.125), but the plot shows only 0.22 ± 0.22.

**Planned:** in the paper figures, show each seed's value as a dot next to the mean, with
diverged and degenerate runs marked differently.

**Why:** makes bimodal outcomes and broken runs visible at a glance.

**Affects:** figures only; no experiment needs to be rerun.
