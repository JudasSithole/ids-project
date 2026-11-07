# Lightweight Behavioural Intrusion Detection for Resource-Constrained Networks

Design and evaluation of a **lightweight behavioural intrusion detection system (IDS)** for resource-constrained networks (e.g., IoT / Raspberry Pi–class devices), benchmarked against **Snort 3 (signature-based)** on CICIDS2017-derived data.

> **Key result (Friday held-out test subset, n=123,161 flows):** tuned **XGBoost** achieved **F1 = 0.9981** (Precision 0.9970, Recall 0.9993, FPR 0.00169). **Random Forest** achieved **F1 = 0.9962**, **Small MLP** **F1 = 0.9822**. **Snort 3.12.2.0 + 4,026 Community Rules** (correlated-flow evaluation, offset +61,260 s, ±300 s window) achieved **F1 = 0.5038** (Precision 0.6906, Recall 0.3966, FPR 0.09881). Under an **emulated 1-core** setting, ML batch inference was efficient (XGB ~134k flows/s, ~333 MiB peak RSS; MLP ~542k flows/s, ~329 MiB); Snort processed the 8.84 GB Friday PCAP offline in **595.44 s (14.16 MB/s)** with ~**200.59 MiB** sampled peak RSS.

## Project goals
- Build a reproducible ML pipeline for **binary intrusion detection** (BENIGN vs ATTACK) using **CICIDS2017 flow features**.
- Train/evaluate **Random Forest, XGBoost, and a small Neural Network (MLP)**.
- Benchmark against **Snort 3** (signature-based) on the **same Friday traffic context** (fair subset comparison).
- Report **Detection Performance** (Accuracy, Precision, Recall, F1, FPR) and **Resource Efficiency** (latency/throughput, CPU, RAM) with **emulated low-spec constraints** (1 core / 1 thread), suitable for Raspberry Pi–class feasibility discussion.

## Repository layout
```text
.
├── README.md
├── requirements.txt
├── snort_friday_alert.lua            # Snort 3 offline config (alert_fast)
├── src/                              # Pipeline, training, evaluation, benchmarking, plots
├── reports/                          # JSON/TXT reports, figures (PNG/PDF), paper drafts
├── data/                             # NOT committed (large). Expected processed artifacts live here.
├── models/                           # NOT committed (large). Trained models/exports live here.
├── snort_rules/                      # Rules/config helpers (community rules content may be large/local)
└── snort_out/                        # NOT committed (large Snort outputs)
```

## What *is* committed vs *not* committed
See `.gitignore`. Large/local artifacts are excluded:
- `data/` (datasets, processed pickles/NPZ, PCAPs), `models/` (joblib/ubj/onnx/npz), `snort_out/`, large parsed alert JSONs (`reports/snort_alerts*.json`), `.venv/`, caches, secrets.
- **Committed**: source code (`src/`), configs, requirements, **reports/metrics JSON summaries**, **figures** (`reports/figures/*.png|*.pdf`), paper drafts (`reports/Project_Report_Draft_IEEE.md`, `reports/Project_Report_IEEE_Final.docx`), and benchmark summaries.

## Environment
- **OS used**: Linux (Ubuntu-based VM in this workspace)
- **Python**: 3.12.x (venv)
- Key libs: `scikit-learn 1.9.1`, `xgboost 3.4.1`, `numpy`, `pandas`, `joblib`, `psutil`, `matplotlib`, `seaborn`, `onnx/onnxruntime/skl2onnx` (for ONNX export/benchmarks)

### Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data (CICIDS2017)
This project uses **CICIDS2017** flow CSVs / generated labelled flows (Friday traffic used for Snort comparison). Due to size/licensing/redistribution, **raw data and large processed files are not committed**.

Expected local artifacts (examples used by scripts):
- `data/processed/cleaned_flows.pkl` — cleaned flow dataset
- `data/processed/split_indices.npz` — stratified 60/20/20 train/val/test indices (seed 42)
- `reports/split_report.json`, `reports/cleaning_report.json`, `reports/dataset_*`
- Friday labelled flow CSVs (e.g., under `/media/.../GeneratedLabelledFlows/TrafficLabelling/` or local equivalent)
- Friday PCAP for Snort offline analysis: `Friday-WorkingHours.pcap` (large; not committed)

> If you do not have the processed artifacts, run the preprocessing/audit/split scripts in `src/` (see below) after placing CICIDS2017 inputs locally.

## Pipeline (high level)
1. **Audit/inspect/clean**: `src/audit_dataset.py`, `src/inspect_dataset.py`, `src/clean_dataset.py`
2. **Split**: `src/create_splits.py` → `data/processed/split_indices.npz`
3. **Train models**: `src/train_random_forest.py`, `src/train_xgboost.py`, `src/train_neural_network.py`
4. **Evaluate (test/tuning/Friday subset)**: `src/evaluate_test.py` and related scripts; reports in `reports/`
5. **Constrained benchmarks (emulated 1 core/1 thread)**: `src/benchmark_constrained.py`, `src/benchmark_models.py`
6. **ONNX export/quantization (optional)**: `src/export_onnx.py` → `models/*.onnx` (not committed)
7. **Snort 3 offline run + correlation**:
   - Run Snort on Friday PCAP with `snort_friday_alert.lua` (produces `alert_fast` output; large, not committed)
   - Parse alerts: `src/parse_snort_alert_fast.py`
   - Correlate alerts to Friday test flows: `src/correlate_snort_friday_fast.py`
   - Snort resource benchmark: `src/benchmark_constrained.py`? (use `src/benchmark_snort_constrained.py`)
8. **Plots / paper figures**: `src/plot_ieee_figures.py` → `reports/figures/01*–07*.{png,pdf}`

## Reproducing key benchmarks

### ML constrained (emulated Raspberry Pi 4–class, 1 core)
```bash
source .venv/bin/activate
python src/benchmark_constrained.py  # see script args/options in file
```
Outputs: `reports/benchmark_*_constrained_*.json`

### Snort constrained (offline PCAP, 1 core)
Requires **Snort 3** installed (this workspace used `Snort 3.12.2.0`) and a local Friday PCAP.
```bash
source .venv/bin/activate
python src/benchmark_snort_constrained.py \
  --pcap /path/to/Friday-WorkingHours.pcap \
  --config snort_friday_alert.lua \
  --cores 0 --sample-interval 1.0 \
  --log-dir snort_out/resource_bench \
  -o reports/benchmark_snort_constrained_pi4_1core.json
```
Result (recorded): **595.44 s wall, 14.16 MB/s PCAP, 200.59 MiB peak RSS, mean CPU% sum 27.84%, return 0** (emulated 1-core via `taskset`).

### Snort alert correlation (Friday test subset)
```bash
python src/parse_snort_alert_fast.py ...
python src/correlate_snort_friday_fast.py ...  # uses tuple index + timestamp offset/window
```
Canonical correlated-flow result: `reports/snort_friday_evaluation_final.json` (**+61,260 s offset, ±300 s window**).

## Results summary

### Detection performance (Friday held-out test subset, binary, n=123,161)
| System | Precision | Recall | F1 | FPR |
|---|---:|---:|---:|---:|
| RF (τ=0.35) | 0.9939 | 0.9986 | **0.9962** | 0.00342 |
| XGBoost (τ=0.70) | 0.9970 | 0.9993 | **0.9981** | 0.00169 |
| Small NN / MLP (τ=0.70) | 0.9953 | 0.9695 | **0.9822** | 0.00255 |
| **Snort 3 + Community Rules (correlated)** | 0.6906 | 0.3966 | **0.5038** | 0.09881 |

### Resource efficiency (emulated 1-core)
| System | Throughput | Latency/flow | Peak RSS | Notes |
|---|---:|---:|---:|---|
| RF | 54,767 flows/s | 0.0183 ms | 351.9 MiB | Batch, 123,161 flows |
| XGBoost | 134,023 flows/s | 0.00746 ms | 332.5 MiB | Batch |
| Small NN (MLP) | 542,020 flows/s | 0.00184 ms | 329.4 MiB | Batch |
| Snort 3.12.2.0 | 14.16 MB/s (PCAP) | N/A | 200.59 MiB | Offline full Friday PCAP 8.84 GB, 595.44 s |

**Interpretation caveat:** Snort is packet/signature processing over a PCAP; ML metrics above are flow-feature batch inference. Correlated-flow Snort evaluation is an approximation (timestamp offsets, 5-tuple mapping, decoder/L2 alerts without tuples). Resource metrics are **emulated** (not physical Raspberry Pi hardware).

## Reports & paper
- **IEEE-style draft (Markdown)**: `reports/Project_Report_Draft_IEEE.md`
- **Filled DOCX (with Fig. 7 embedded)**: `reports/Project_Report_IEEE_Final.docx`
- **Aggregate summary**: `reports/final_comparison_summary.json`
- **Snort correlation (final)**: `reports/snort_friday_evaluation_final.json`
- **Snort resource**: `reports/benchmark_snort_constrained_pi4_1core.json`
- **Figures (publication, IEEE double-column, PNG+PDF)**: `reports/figures/01*–07*`
  - Fig. 7: `reports/figures/07_snort_ml_comparison.{png,pdf}` (ML vs Snort 3)

## Reproducibility notes
- Stratified split: 60/20/20, **seed 42** (`reports/split_report.json`, `data/processed/split_indices.npz`).
- ML constrained benches: 1-core affinity + single-thread BLAS/OpenMP/threadpool controls; sampled via `psutil`.
- Snort constrained bench: `taskset -c 0`, offline `-r` + `-A alert_fast -q`; throughput reported as **PCAP MB/s**.
- Large data/models/PCAPs/Snort outputs are intentionally excluded from git.

## Limitations & future work
- Friday-only Snort PCAP comparison; extend to other CICIDS2017 days/PCAPs.
- Packet-alert ↔ flow-label correlation is imperfect (aggregation, time alignment, non-5-tuple alerts).
- Emulated low-spec, not physical Raspberry Pi/IoT deployment.
- Future: improved correlation, alternative rulesets (e.g., ET), explainability (SHAP), hybrid signature+ML, on-device evaluation.

## License / acknowledgements
- **CICIDS2017**: Canadian Institute for Cybersecurity / University of New Brunswick (see dataset terms).
- **Snort**: Cisco/Snort project; Community Rules are distributed under their respective licenses (kept under `snort_rules/` as applicable).
- This repository contains research code and report artifacts for academic evaluation; verify third-party licenses before redistribution.
