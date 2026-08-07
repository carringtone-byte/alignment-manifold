# Result Snapshots

These JSON files are compact copies of the machine-readable reports generated
by the completed 7B experiment. They were copied from the ignored experiment
artifact directory so that the public claims remain inspectable without
shipping model weights or activation arrays.

| File | Contents |
| --- | --- |
| `trajectory_7b_rank32.json` | Preregistered rank-32 trajectory analysis |
| `trajectory_7b_extended_rank.json` | Exploratory extended-rank analysis |
| `trajectory_7b_robustness.json` | Split and category robustness analyses |
| `trajectory_7b_causal.json` | Causal trajectory evaluation |
| `provenance/*.manifest.json` | Resolved checkpoint and extraction provenance |

The primary prose interpretation is in
`../reports/TRAJECTORY_7B_RESULTS.md`. Large `.npz` activation arrays are not
included.

