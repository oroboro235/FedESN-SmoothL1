# gen_rounds_data.py — per-round accuracy AND sparsity history for the
# federated convergence figure (accuracy | sparsity vs communication round).
#
# Runs the federated SL1 protocol (K=5, 50 rounds, tau=0.1) for each dataset
# over ten seeds and records the per-round test accuracy and readout sparsity
# from tsc_fl_eval's round_history, then aggregates mean/std across seeds.
# Output: result/pic_10seed/rounds_evolution.json, consumed by plot_paper_figs.py.
#
# Usage:  python gen_rounds_data.py

import json
import os
from copy import deepcopy

import numpy as np

import config
from tsc_fl_eval import main as run_experiment

ORDER = ["har", "char", "jpv", "ECG5000",
         "DistalPhalanxOutlineCorrect", "Yoga", "Strawberry"]
# Seeds 92..101 reproduce the tau=0.1 column of the threshold sweep behind
# Table 5, so the round-50 endpoint of this figure equals the table value.
N_SEEDS   = 10
BASE_SEED = 92
N_ROUNDS  = 50
N_CLIENTS = 5
TAU       = 0.1
OUT = "./result/pic_10seed/rounds_evolution.json"


def _base(ds):
    path = os.path.join(config.paths.configs_path, "TSC_FL_settings_sl1.json")
    for i, e in enumerate(json.load(open(path))):
        if e["dataset"] == ds:
            return deepcopy(e), i
    raise KeyError(ds)


def main():
    data = {}
    for ds in ORDER:
        base, idx = _base(ds)
        overrides = {
            "sr":               base["sr"],
            "lr":               base["lr"],
            "input_scaling":    base["input_scaling"],
            "reg_param":        base["reg_param"],
            "thres":            TAU,
            "alpha_init":       1.0,
            "alpha_multiplier": 2.0,
        }
        accs, sps, rounds = [], [], None
        for s in range(N_SEEDS):
            res = run_experiment(
                reg_type="sl1", n_rounds=N_ROUNDS, n_clients=N_CLIENTS,
                seed=BASE_SEED + s, thres=TAU, setting_idx=idx,
                datasets=[ds], param_overrides=overrides,
                patience=0, verbose=False,
            )
            rh = res[ds]["round_history"]
            accs.append([r["test_acc"] for r in rh])
            sps.append([r["sparsity"] for r in rh])
            rounds = [r["round"] for r in rh]
        acc = np.asarray(accs)
        sp  = np.asarray(sps)
        data[ds] = {
            "rounds":   rounds,
            "acc_mean": acc.mean(0).tolist(), "acc_std": acc.std(0).tolist(),
            "sp_mean":  sp.mean(0).tolist(),  "sp_std":  sp.std(0).tolist(),
        }
        print(f"{ds:30} final acc {acc.mean(0)[-1]:5.1f}  sparsity {sp.mean(0)[-1]:5.1f}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(data, open(OUT, "w"))
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
