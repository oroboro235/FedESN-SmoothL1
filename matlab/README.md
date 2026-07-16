# MATLAB sensitivity plotting

Reproduce the sensitivity figures from the JSON exports written by the Python
sweep scripts (`tsc_centralized_sensitivity.py` / `tsc_fl_sensitivity.py`,
`save_run()`).

Each Python run writes, alongside its PNG/PDF, a JSON file into `--out_dir`
(default `result/pic/`):

- centralized: `sensitivity_<dataset>_<param>.json`
- federated:   `fl_sensitivity_<dataset>_<reg>_<param>.json`

Each JSON holds every hyperparameter of the run (`.settings`) plus the per-value
results (`.results`), and axis-formatting hints (`.is_log`, `.is_integer`).

## Usage

```matlab
cd matlab

% One figure from one export (saves PNG+PDF next to the JSON):
plot_sensitivity('../result/pic/fl_sensitivity_har_sl1_n_clients.json');

% ...or into a chosen output dir:
plot_sensitivity('../result/pic/sensitivity_har_reg_param.json', 'figs');

% Batch: plot every *sensitivity_*.json in a directory (default ../result/pic):
plot_all_sensitivity();
plot_all_sensitivity('../result/pic', 'figs');
```

Figures use a dual y-axis (accuracy left, sparsity right), both fixed to
`[0 100]`. Log-scaled params (`reg_param`, `thres`, `input_scaling`) get a log
x-axis; integer params (`n_clients`, `n_rounds`, `units`, `epochs`) get integer
x-ticks only. Requires MATLAB R2020a+ (`exportgraphics`, `jsondecode`, `xline`).
