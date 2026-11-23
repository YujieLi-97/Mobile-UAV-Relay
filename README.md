# Online radio-map guidance for mobile UAV relay trajectories

This repository contains the single-file simulation code, manuscript source, and seed-level results for the paired synthetic-simulation study of a moving two-hop UAV relay. The 3.0-Mbit/s controller settings were selected on development seeds 3000–3099 and locked before the original 200-scene holdout evaluation on seeds 4000–4199. Reviewer follow-up analyses were added later and are labeled retrospective or exploratory in the manuscript.

Numerical outputs use the `.dat` extension. Metric and summary files contain UTF-8 JSON text and can be read with `json.load`; `*.series.dat` files contain NumPy compressed arrays and can be read with `numpy.load`.

## Run

Python 3.12 is recommended. From this folder:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -q
.venv/bin/python uav_relay_experiment.py verify
```

The complete original campaign can be reproduced with:

```bash
.venv/bin/python uav_relay_experiment.py all --workers 8
```

Reviewer follow-up runs use the same file:

```bash
.venv/bin/python uav_relay_experiment.py review-checks --workers 8
.venv/bin/python uav_relay_experiment.py feedback-holdout --workers 8
.venv/bin/python uav_relay_experiment.py robustness --workers 8
.venv/bin/python uav_relay_experiment.py risk-audit --workers 8
.venv/bin/python uav_relay_experiment.py boundary-audit
.venv/bin/python uav_relay_experiment.py boundary-safe --workers 8
.venv/bin/python uav_relay_experiment.py boundary-safe-holdout --workers 8
```

`results_round4/` holds the original campaign records. `results_review/` holds scene-level follow-up records, while `results_review_*.dat` files contain summaries. All numerical outputs have `.dat` names: metric and summary files are UTF-8 structured text; `*.series.dat` files contain NumPy compressed arrays. The manuscript source is `paper/main.tex`, and its compiled PDF is `paper/manuscript.pdf`. The exact experimental selection and analysis rules are documented in `Round4_Protocol.md`.

The `verify` command checks the original campaign's embedded source signature. The boundary-safe and feedback follow-ups are separate post-hoc analyses and do not change that signature. No measured-channel validation or direct reproduction of the cited external controllers is included.

`results_review_manifest.dat` retains the full-file hash recorded when the reviewer-analysis manifest was assembled and also lists the current file hash. `results_review_source_diff.dat` shows the exact change between those versions: a redraw of the system-model figure. The numerical experiment modules and primary run signature are unchanged. The boundary-feasible holdout rerun covers the selected GP policy; the original midpoint path already meets the realized acceleration bound. Other affected comparators have not been rerun with the same feasible-motion rule.

To reconstruct the manifest version, run `patch -R -o /tmp/uav_manifest_recovered.py uav_relay_experiment.py results_review_source_diff.dat`. Its SHA-256 is `09703b7521cd21b95dd112b436a5f7783696162d62de77f7bd6a34c36acd6e8a`; the current file is `cd932d44aae5a28dcd00a063352ac8c7582d52e4d403efacf7d5084f48a2ebd7`.

