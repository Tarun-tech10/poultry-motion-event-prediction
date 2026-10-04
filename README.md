# Poultry motion event prediction

Working code for predicting hen-house motion events across future horizons.
Each 16x30 motion strip is decoded onto a canonical 4x4 spatial grid; per-cell
features (recency, trends, lattice neighbourhoods, optical-flow advection, AR
forecasts) feed gradient-boosted and GNN models that predict the zone and tier
of the next events, and a decoder maximises the grader's expected score.

Key files: `common.py` (strip decoding), `feats*.py`, `folds.py`, `gbm*.py`,
`gnn.py`, `model.py`, `train.py`, `blend.py`, `final_ens.py`, `decode.py`,
`scorer.py`. `*.log` files are the cross-validation runs.

Final solution: https://github.com/Tarun-tech10/poultry-motion-event-solution

The dataset, cached arrays and out-of-fold predictions (`*.npy`) are not included.

Licensed under the MIT License.
