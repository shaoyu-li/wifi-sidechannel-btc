# Bitcoin_Identify_ML — Wi-Fi Bitcoin traffic detection & attacker-centric victim identification

This repository provides a Wi-Fi traffic analysis pipeline that:

1) trains a **per-window multi-class traffic classifier** (XGBoost) using a fixed **50-D feature extractor** computed from encrypted IEEE 802.11 frame metadata; and  
2) runs an **attacker-centric evaluation** that simulates multi-user Wi-Fi scenarios and ranks users by **Bitcoin likelihood** (mean `p(bitcoinsta)`), under increasing background-application mixing levels `k = 0…5`.

The learning algorithm (feature extraction + XGBoost) is fixed; the experiments focus on **Bitcoin detection and victim identification under mixed traffic conditions**.

> **Scope note.**  
> This module performs traffic classification and attacker-centric ranking using *passively observable* Wi-Fi frame metadata only. It does **not** include Internet-side probing, packet injection, or protocol manipulation.

---

## Dataset format (current)

Place the dataset under `data/` as **flat JSON files**:

- Each `data/*.json` file is treated as a **distinct application class**  
  (class ID = filename stem).
- **Bitcoin class (current)**: `bitcoinsta` (`data/bitcoinsta.json`).  
  At present, Bitcoin traffic in this dataset consists **only** of `bitcoinsta`.
- **Background classes (current snapshot)**:  
  `download`, `game`, `music`, `video`, `web`.

Adding additional `data/*.json` files will automatically introduce new
application classes.

Each JSON file must contain a list of Wi-Fi frame records with **at least**
the following fields:

- `ts` — float timestamp  
- `phy_len`, `mac_len`, `enc_payload_len` — int  
- `sa`, `da`, `bssid` — MAC address strings  
- `type`, `subtype` — int  
- `protected`, `retry` — 0 / 1  
- `direction` — "tx" or "rx"  
- `rssi` — int or float  

---

## Environment (recommended)

We recommend using **conda** to avoid platform-specific NumPy / OpenBLAS
issues (especially on macOS):

```bash
conda create -n wifi_btc python=3.10 -y
conda activate wifi_btc
conda install -c conda-forge   numpy pandas scikit-learn xgboost joblib matplotlib -y
```

---

## Repository structure

```text
Bitcoin_Identify_ML/
├── bitcoin_detector.py
├── experiments/
│   ├── __init__.py
│   ├── run.py
│   └── config_example.json
├── data/
│   ├── bitcoinsta.json
│   ├── download.json
│   ├── game.json
│   ├── music.json
│   ├── video.json
│   └── web.json
└── results/
    ├── metrics.csv
    ├── report.md
    └── figures/
        └── attacker_metrics_vs_num_background_apps.png
```

---

## Quick start

### 1) Train the multi-class model (pure windows only)

```bash
python -u bitcoin_detector.py
```

This trains a **multi-class XGBoost** model over `data/*.json`
(pure windows only) and saves:

- `wifi_multiclass_model.pkl`

---

### 2) Attacker-centric victim identification experiment (recommended)

```bash
python -u -m experiments.run --config experiments/config_example.json
```

This simulates scenarios with:

- `N` users (e.g., 5 / 10 / 20)
- mixing level `k` = number of concurrent background application types
  mixed with Bitcoin for Bitcoin users (`k = 0…5`)
- a subset of users running Bitcoin; others background-only

The attacker computes a **per-user score**
(mean `p(bitcoinsta)`), ranks users, and reports
user-level detection and ranking metrics.

---

## Pipeline summary

```text
data/*.json (each file = one class)
  └─> leakage-safe per-class time split (train / test)
      ├─> windowing (200 ms window, 100 ms step, min 10 frames)
      │    └─> 50-D feature extraction
      │         (size / timing / direction / type-subtype /
      │          RSSI / MAC diversity / protected / retry)
      │         └─> XGBoost multi-class training
      │              (pure windows only)
      │
      └─> attacker-centric evaluation (test windows only)
           ├─> simulate N users (MAC remapping)
           ├─> mix windows into blocks per user
           │    - Bitcoin user: bitcoinsta + k background apps
           │    - Non-Bitcoin user: background apps only
           ├─> user score = mean p(bitcoinsta) over blocks
           └─> ranking + metrics + plots under results/
```

---

## Outputs

After running the experiment runner, results are written to:

- `results/metrics.csv`  
  Columns: `seed, N, MixingLevel, metric_name, value`

- `results/figures/attacker_metrics_vs_num_background_apps.png`

- `results/report.md`  
  A concise report summarizing the run and embedding the key figure.

---

## Experiment configuration

Edit `experiments/config_example.json` to control:

### Scenario size
- `N_values`
- `num_runs`
- `seed_start`

### Mixing
- `mixing_levels`  
  (interpreted as “number of background applications”)

### User aggregation
- `user_agg_num_blocks`

### Client simulation stability
- `bitcoin_packet_fraction_in_block`
- `btc_target_packets_per_block`
- `min_component_packets`

### Attacker decision rule
- `decision_rule.type = "topm"`  
  (rank users by score; flag top-m suspects)
- `decision_rule.min_m_pred`

---

## Reproducibility notes

- The experiment runner applies a **leakage-safe split**:
  per-class **time split** into train/test before windowing.
- For consistent results, keep dependency versions fixed
  (conda environment above) and avoid modifying
  `config_example.json` unless necessary.
- Exact numerical results depend on the dataset snapshot under `data/`
  and the configured random seeds.

---

## Contact / issues

If you encounter issues reproducing results, please include:

- OS version, Python version, and package versions
- the configuration JSON used for the run
- the full command line and console logs
