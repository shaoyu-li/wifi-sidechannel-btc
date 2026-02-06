import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from bitcoin_detector import (
    bitcoinsta_score_fn,
    build_train_test_segments_from_classes,
    extract_features,
    infer_shared_bssid,
    load_data_classes,
    remap_component_for_user,
    train_multiclass_xgb_from_segments,
)


def _mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _load_config(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def _pick_k_distinct(rng: np.random.Generator, items: List[str], k: int) -> List[str]:
    if k <= 0:
        return []
    k = min(k, len(items))
    idx = rng.choice(len(items), size=k, replace=False)
    return [items[i] for i in idx.tolist()]


def _sample_window(rng: np.random.Generator, pool: List[List[Dict]]) -> List[Dict]:
    if not pool:
        raise RuntimeError("Empty window pool encountered.")
    return pool[int(rng.integers(0, len(pool)))]


def _sample_window_min_len(
    rng: np.random.Generator, pool: List[List[Dict]], min_len: int, max_tries: int = 50
) -> List[Dict]:
    if not pool:
        raise RuntimeError("Empty window pool encountered.")
    min_len = max(1, int(min_len))
    for _ in range(max_tries):
        w = _sample_window(rng, pool)
        if len(w) >= min_len:
            return w
    # Fallback: return whatever we got
    return _sample_window(rng, pool)


def _downsample_packets_keep_order(
    rng: np.random.Generator, packets: List[Dict], target_n: int
) -> List[Dict]:
    n = len(packets)
    target_n = int(target_n)
    if target_n <= 0:
        return []
    if n <= target_n:
        return packets
    idx = rng.choice(n, size=target_n, replace=False)
    idx_sorted = np.sort(idx)
    return [packets[int(i)] for i in idx_sorted.tolist()]


def _normalize_block_time(packets: List[Dict], window_duration: float) -> List[Dict]:
    if not packets:
        return packets
    ts_vals = [float(p.get("ts", 0.0)) for p in packets]
    tmax = max(ts_vals)
    tmin = min(ts_vals)
    if tmax <= tmin:
        # Collapse to 0
        return [dict(p, ts=0.0) for p in packets]
    scale = float(window_duration) * 0.999 / (tmax - tmin)
    out = [dict(p, ts=(float(p.get("ts", 0.0)) - tmin) * scale) for p in packets]
    out.sort(key=lambda x: x.get("ts", 0))
    return out


def generate_user_blocks(
    *,
    N: int,
    mixing_level: int,
    bitcoin_users_per_scenario: int,
    segments_pool: Dict[str, List[List[Dict]]],
    class_names: List[str],
    bitcoinsta_class: str,
    shared_bssid: str,
    user_agg_num_blocks: int,
    window_duration: float,
    mix_time_jitter_s: float,
    bitcoin_packet_fraction_in_block: float,
    btc_target_packets_per_block: int,
    min_component_packets: int,
    seed: int,
) -> Tuple[List[List[List[Dict]]], np.ndarray]:
    """
    Return user_blocks[user][block] -> packets, plus labels[user] in {0,1}.
    """
    rng = np.random.default_rng(seed)
    non_btc = [c for c in class_names if c != bitcoinsta_class]
    btc_users = set(rng.choice(N, size=bitcoin_users_per_scenario, replace=False).tolist())
    labels = np.zeros(N, dtype=int)
    for u in btc_users:
        labels[u] = 1

    user_blocks: List[List[List[Dict]]] = []
    for u in range(N):
        if labels[u] == 1:
            others = _pick_k_distinct(rng, non_btc, mixing_level)
            user_classes = [bitcoinsta_class] + others
        else:
            m = min(mixing_level + 1, len(non_btc))
            user_classes = _pick_k_distinct(rng, non_btc, m)

        # Dedicated RNG so each user gets stable block sampling
        block_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
        blocks: List[List[Dict]] = []
        for b in range(user_agg_num_blocks):
            # Packet budgeting to keep Bitcoin signal stable as k grows.
            # We target a fixed number of bitcoin packets, and downsample background packets
            # so that bitcoin_fraction_in_block stays roughly constant.
            btc_frac = float(bitcoin_packet_fraction_in_block)
            btc_frac = min(max(btc_frac, 0.05), 0.95)
            btc_target = int(max(min_component_packets, btc_target_packets_per_block))
            total_target = int(max(btc_target, round(btc_target / btc_frac)))

            comps: List[List[Dict]] = []

            if labels[u] == 1:
                # Build bitcoin component to hit btc_target.
                btc_packets: List[Dict] = []
                # Concatenate multiple windows if needed (still same algorithm; just more btc traffic).
                while len(btc_packets) < btc_target:
                    seg_btc = _sample_window_min_len(block_rng, segments_pool[bitcoinsta_class], min_len=min_component_packets)
                    rem = remap_component_for_user(
                        seg_btc,
                        user_idx=u,
                        shared_bssid=shared_bssid,
                        time_jitter_s=mix_time_jitter_s,
                        rng=block_rng,  # type: ignore[arg-type]
                    )
                    btc_packets.extend(rem)
                    if len(btc_packets) > btc_target * 3:
                        break
                btc_packets = _downsample_packets_keep_order(block_rng, btc_packets, btc_target)
                comps.append(btc_packets)

                bg_classes = [cn for cn in user_classes if cn != bitcoinsta_class]
                bg_budget = max(0, total_target - len(btc_packets))
                if bg_classes and bg_budget > 0:
                    # Split budget across background apps.
                    per_bg = max(min_component_packets, bg_budget // len(bg_classes))
                    for cn in bg_classes:
                        seg = _sample_window_min_len(block_rng, segments_pool[cn], min_len=min_component_packets)
                        rem = remap_component_for_user(
                            seg,
                            user_idx=u,
                            shared_bssid=shared_bssid,
                            time_jitter_s=mix_time_jitter_s,
                            rng=block_rng,  # type: ignore[arg-type]
                        )
                        rem = _downsample_packets_keep_order(block_rng, rem, per_bg)
                        comps.append(rem)
            else:
                # Negative user: fill total_target using only non-bitcoin classes.
                bg_classes = user_classes
                if not bg_classes:
                    bg_classes = _pick_k_distinct(block_rng, non_btc, 1)
                per_bg = max(min_component_packets, total_target // max(1, len(bg_classes)))
                for cn in bg_classes:
                    seg = _sample_window_min_len(block_rng, segments_pool[cn], min_len=min_component_packets)
                    rem = remap_component_for_user(
                        seg,
                        user_idx=u,
                        shared_bssid=shared_bssid,
                        time_jitter_s=mix_time_jitter_s,
                        rng=block_rng,  # type: ignore[arg-type]
                    )
                    rem = _downsample_packets_keep_order(block_rng, rem, per_bg)
                    comps.append(rem)

            mixed = [p for comp in comps for p in comp]
            mixed.sort(key=lambda x: x.get("ts", 0))
            mixed = _normalize_block_time(mixed, window_duration)

            # Offset block times so blocks are non-overlapping in the per-user stream.
            t_off = b * window_duration
            if t_off != 0.0:
                mixed = [dict(p, ts=float(p.get("ts", 0.0)) + t_off) for p in mixed]

            blocks.append(mixed)
        user_blocks.append(blocks)

    return user_blocks, labels


def compute_user_scores_from_blocks(
    user_blocks: List[List[List[Dict]]],
    score_fn,
) -> np.ndarray:
    scores = np.zeros(len(user_blocks), dtype=float)
    for u, blocks in enumerate(user_blocks):
        block_scores = [score_fn(b) for b in blocks]
        scores[u] = float(np.mean(block_scores)) if block_scores else 0.0
    return scores


def compute_user_scores_from_blocks_batched(
    user_blocks: List[List[List[Dict]]],
    *,
    model,
    bitcoinsta_idx: int,
) -> np.ndarray:
    """Same scoring, but batches model inference for speed."""
    # Flatten blocks
    feats = []
    user_slices: List[Tuple[int, int]] = []
    for blocks in user_blocks:
        start = len(feats)
        for b in blocks:
            feats.append(extract_features(b))
        end = len(feats)
        user_slices.append((start, end))

    if not feats:
        return np.zeros(len(user_blocks), dtype=float)

    X = np.vstack(feats)
    proba = model.predict_proba(X)[:, bitcoinsta_idx]

    scores = np.zeros(len(user_blocks), dtype=float)
    for u, (a, b) in enumerate(user_slices):
        scores[u] = float(np.mean(proba[a:b])) if b > a else 0.0
    return scores


def select_tau_on_train_scenarios(
    *,
    model,
    bitcoinsta_idx: int,
    segments_pool: Dict[str, List[List[Dict]]],
    class_names: List[str],
    bitcoinsta_class: str,
    shared_bssid: str,
    N_values: List[int],
    mixing_levels: List[int],
    bitcoin_users_per_scenario: int,
    user_agg_num_blocks: int,
    window_duration: float,
    mix_time_jitter_s: float,
    bitcoin_packet_fraction_in_block: float,
    btc_target_packets_per_block: int,
    min_component_packets: int,
    num_val_scenarios_per_setting: int,
    tau_grid_size: int,
    seed: int,
) -> Dict[Tuple[int, int], float]:
    """Calibrate tau separately for each (N, MixingLevel) using train-only scenarios."""
    tau_table: Dict[Tuple[int, int], float] = {}
    run_id = 0
    for N in N_values:
        for k in mixing_levels:
            scores_all: List[float] = []
            labels_all: List[int] = []
            for r in range(num_val_scenarios_per_setting):
                run_seed = seed * 100000 + N * 1000 + k * 100 + r
                user_blocks, labels = generate_user_blocks(
                    N=N,
                    mixing_level=k,
                    bitcoin_users_per_scenario=bitcoin_users_per_scenario,
                    segments_pool=segments_pool,
                    class_names=class_names,
                    bitcoinsta_class=bitcoinsta_class,
                    shared_bssid=shared_bssid,
                    user_agg_num_blocks=user_agg_num_blocks,
                    window_duration=window_duration,
                    mix_time_jitter_s=mix_time_jitter_s,
                    bitcoin_packet_fraction_in_block=bitcoin_packet_fraction_in_block,
                    btc_target_packets_per_block=btc_target_packets_per_block,
                    min_component_packets=min_component_packets,
                    seed=run_seed,
                )
                scores = compute_user_scores_from_blocks_batched(
                    user_blocks, model=model, bitcoinsta_idx=bitcoinsta_idx
                )
                scores_all.extend(scores.tolist())
                labels_all.extend(labels.tolist())
                run_id += 1

            s = np.array(scores_all, dtype=float)
            y = np.array(labels_all, dtype=int)
            # Candidate taus from score quantiles (more relevant than [0,1] grid).
            qs = np.linspace(0.0, 1.0, int(tau_grid_size))
            candidates = np.unique(np.quantile(s, qs))
            best_tau = float(np.median(s))
            best_f1 = -1.0
            for tau in candidates:
                pred = (s >= tau).astype(int)
                _, _, f1, _ = precision_recall_fscore_support(
                    y, pred, average="binary", zero_division=0
                )
                if float(f1) > best_f1:
                    best_f1 = float(f1)
                    best_tau = float(tau)

            tau_table[(int(N), int(k))] = best_tau
            print(f"  Calibrated tau for N={N}, k={k}: tau={best_tau:.6f} (best F1={best_f1:.4f})")

    return tau_table


def compute_metrics_one_scenario(scores: np.ndarray, labels: np.ndarray, tau: float, topk_list=(1, 3, 5)) -> Dict[str, float]:
    pred = (scores >= tau).astype(int)
    acc = float((pred == labels).mean()) if len(labels) else 0.0
    precision, recall, f1, _ = precision_recall_fscore_support(labels, pred, average="binary", zero_division=0)

    # AUCs
    roc_auc = roc_auc_score(labels, scores)
    pr_auc = average_precision_score(labels, scores)

    # Top-K
    order = np.argsort(-scores)  # descending
    pos_idx = set(np.where(labels == 1)[0].tolist())
    topk = {}
    for K in topk_list:
        K_eff = min(int(K), len(scores))
        top = set(order[:K_eff].tolist())
        topk[f"top{K}"] = 1.0 if len(top & pos_idx) > 0 else 0.0

    # False positives
    neg_mask = labels == 0
    fp_scenario = float(((pred == 1) & neg_mask).sum())
    fp_per_user = fp_scenario / float(max(1, int(neg_mask.sum())))

    out = {
        "acc": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "fp_per_scenario": fp_scenario,
        "fp_per_user": float(fp_per_user),
        **topk,
    }
    return out


def compute_metrics_one_scenario_topm(
    scores: np.ndarray,
    labels: np.ndarray,
    m_pred: int,
    topk_list=(1, 3, 5),
) -> Dict[str, float]:
    n = len(scores)
    m_pred = int(max(1, min(m_pred, n)))
    order = np.argsort(-scores)
    pred = np.zeros(n, dtype=int)
    pred[order[:m_pred]] = 1

    acc = float((pred == labels).mean()) if len(labels) else 0.0
    precision, recall, f1, _ = precision_recall_fscore_support(labels, pred, average="binary", zero_division=0)
    roc_auc = roc_auc_score(labels, scores)
    pr_auc = average_precision_score(labels, scores)

    pos_idx = set(np.where(labels == 1)[0].tolist())
    topk = {}
    for K in topk_list:
        K_eff = min(int(K), n)
        top = set(order[:K_eff].tolist())
        topk[f"top{K}"] = 1.0 if len(top & pos_idx) > 0 else 0.0

    neg_mask = labels == 0
    fp_scenario = float(((pred == 1) & neg_mask).sum())
    fp_per_user = fp_scenario / float(max(1, int(neg_mask.sum())))

    return {
        "acc": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "fp_per_scenario": float(fp_scenario),
        "fp_per_user": float(fp_per_user),
        **topk,
    }


def summarize_metrics(df: pd.DataFrame, metric_names: List[str]) -> pd.DataFrame:
    rows = []
    for (N, k, metric), g in df.groupby(["N", "MixingLevel", "metric_name"]):
        vals = g["value"].astype(float).values
        rows.append(
            {
                "N": int(N),
                "MixingLevel": int(k),
                "metric_name": metric,
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "n": int(len(vals)),
            }
        )
    out = pd.DataFrame(rows)
    # Keep only requested metrics if provided
    if metric_names:
        out = out[out["metric_name"].isin(metric_names)]
    return out.sort_values(["metric_name", "N", "MixingLevel"]).reset_index(drop=True)


def make_plots(summary_df: pd.DataFrame, results_fig_dir: Path) -> None:
    import matplotlib.pyplot as plt

    # Requested: no F1; plot only acc/top1/top3 vs number of background apps (k)
    metrics_to_plot = ["acc", "top1", "top3"]
    Ns = sorted(summary_df["N"].unique().tolist())
    ks = sorted(summary_df["MixingLevel"].unique().tolist())

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True, sharey=False)
    for ax, metric in zip(axes, metrics_to_plot):
        for N in Ns:
            sub = summary_df[(summary_df["metric_name"] == metric) & (summary_df["N"] == N)]
            y = []
            for k in ks:
                row = sub[sub["MixingLevel"] == k]
                y.append(float(row["mean"].iloc[0]) if len(row) else np.nan)
            ax.plot(ks, y, marker="o", label=f"N={N}")
        ax.set_title(metric)
        ax.set_xlabel("Num background apps (k)")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    out_path = results_fig_dir / "attacker_metrics_vs_num_background_apps.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def write_report(
    *,
    report_path: Path,
    config: Dict,
    summary_df: pd.DataFrame,
    figures_rel_paths: List[str],
) -> None:
    lines: List[str] = []
    lines.append("# Attacker-centric victim identification report\n")
    lines.append("## Experiment description\n")
    lines.append(
        "We train the existing per-window Wi‑Fi traffic classifier (feature extraction unchanged) using **pure** windows from each `data/*.json` class. "
        "We then evaluate an attacker that ranks users by their **Bitcoin likelihood** (`p(bitcoinsta)`), under simulated N-user mixed-traffic Wi‑Fi scenarios.\n"
    )
    lines.append("In each scenario, there are N users. A subset runs Bitcoin (`bitcoinsta`) mixed with `k` concurrent non-Bitcoin application types (MixingLevel). Others run only non-Bitcoin apps.\n")

    lines.append("## Parameters\n")
    param_rows = [
        ("N_values", config["N_values"]),
        ("mixing_levels", config["mixing_levels"]),
        ("num_runs", config["num_runs"]),
        ("seed_start", config["seed_start"]),
        ("bitcoin_users_per_scenario", config["bitcoin_users_per_scenario"]),
        ("window_duration_s", config["window_duration"]),
        ("user_agg_num_blocks", config["user_agg_num_blocks"]),
        ("decision_rule", "top-m users per scenario (m=round(alpha*N)), alpha calibrated per-(N,k) on train-only scenarios"),
    ]
    lines.append("| Parameter | Value |\n|---|---|\n")
    for k, v in param_rows:
        lines.append(f"| `{k}` | `{v}` |\n")

    lines.append("\n## Metrics summary (mean±std over runs)\n")
    for N in sorted(summary_df["N"].unique().tolist()):
        lines.append(f"\n### N = {N}\n")
        sub = summary_df[summary_df["N"] == N].copy()
        # Pivot to one table per metric group by k
        ks = sorted(sub["MixingLevel"].unique().tolist())
        metrics = ["f1", "pr_auc", "top1", "top3", "roc_auc", "precision", "recall", "fp_per_scenario", "fp_per_user"]
        lines.append("| k | " + " | ".join(metrics) + " |\n")
        lines.append("|---|" + "|".join(["---"] * len(metrics)) + "|\n")
        for k in ks:
            row_parts = [str(k)]
            for m in metrics:
                r = sub[(sub["MixingLevel"] == k) & (sub["metric_name"] == m)]
                if len(r) == 0:
                    row_parts.append("NA")
                else:
                    mu = float(r["mean"].iloc[0])
                    sd = float(r["std"].iloc[0])
                    row_parts.append(f"{mu:.4f}±{sd:.4f}")
            lines.append("| " + " | ".join(row_parts) + " |\n")

    if figures_rel_paths:
        lines.append("\n## Figures\n")
        for rel in figures_rel_paths:
            lines.append(f"\n![]({rel})\n")

    report_path.write_text("".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="Path to JSON config file")
    args = ap.parse_args()

    config = _load_config(args.config)
    seed_start = int(config["seed_start"])
    num_runs = int(config["num_runs"])
    seeds = list(range(seed_start, seed_start + num_runs))

    N_values = [int(x) for x in config["N_values"]]
    mixing_levels = [int(x) for x in config["mixing_levels"]]
    bitcoin_users_per_scenario = int(config["bitcoin_users_per_scenario"])

    data_root = str(config["data_root"])
    bitcoinsta_class = str(config["bitcoinsta_class"])
    train_frac = float(config["train_frac"])

    window_duration = float(config["window_duration"])
    step_size = float(config["step_size"])
    min_frames = int(config["min_frames"])
    max_windows_train_per_class = int(config["max_windows_train_per_class"])
    max_windows_test_per_class = int(config["max_windows_test_per_class"])
    max_train_windows_per_class_for_model = int(config["max_train_windows_per_class_for_model"])

    user_agg_num_blocks = int(config["user_agg_num_blocks"])
    mix_time_jitter_s = float(config["mix_time_jitter_s"])
    bitcoin_packet_fraction_in_block = float(config.get("bitcoin_packet_fraction_in_block", 0.4))
    btc_target_packets_per_block = int(config.get("btc_target_packets_per_block", 400))
    min_component_packets = int(config.get("min_component_packets", max(10, min_frames)))

    thr_cfg = config.get("threshold_selection", {})
    num_val_scenarios_per_setting = int(thr_cfg.get("num_val_scenarios_per_setting", 5))
    tau_grid_size = int(thr_cfg.get("tau_grid_size", 81))

    dr_cfg = config.get("decision_rule", {})
    decision_type = str(dr_cfg.get("type", "topm"))
    alpha_grid_size = int(dr_cfg.get("alpha_grid_size", 31))
    min_m_pred = int(dr_cfg.get("min_m_pred", 1))
    min_m_pred_apply_N_ge = int(dr_cfg.get("min_m_pred_apply_N_ge", 0))
    if decision_type not in ("topm",):
        raise ValueError(f"Unsupported decision_rule.type: {decision_type}")

    results_dir = Path(str(config.get("results_dir", "results")))
    fig_dir = results_dir / "figures"
    _mkdir(results_dir)
    _mkdir(fig_dir)

    print("=" * 72)
    print("ATTACKER-CENTRIC VICTIM IDENTIFICATION EXPERIMENT")
    print("=" * 72)

    print("\n[STAGE 1/5] Loading data classes...")
    classes_packets = load_data_classes(data_root)
    class_names = sorted(classes_packets.keys())
    print(f"  Found {len(class_names)} classes: {class_names}")
    if bitcoinsta_class not in classes_packets:
        raise RuntimeError(f"Bitcoin class not found: {bitcoinsta_class}.json")

    shared_bssid = infer_shared_bssid(classes_packets[bitcoinsta_class])
    if shared_bssid is None:
        raise RuntimeError("Could not infer shared BSSID from bitcoinsta packets.")
    print(f"  Using shared BSSID: {shared_bssid}")

    print("\n[STAGE 2/5] Building train/test segments (leakage-free time split)...")
    train_segments, test_segments = build_train_test_segments_from_classes(
        classes_packets,
        train_frac=train_frac,
        window_duration=window_duration,
        step_size=step_size,
        min_frames=min_frames,
        max_windows_train_per_class=max_windows_train_per_class,
        max_windows_test_per_class=max_windows_test_per_class,
    )
    for cn in class_names:
        print(f"  {cn}: train_windows={len(train_segments[cn])}, test_windows={len(test_segments[cn])}")

    print("\n[STAGE 3/5] Training multi-class XGBoost (algorithm unchanged)...")
    model, class_to_idx = train_multiclass_xgb_from_segments(
        train_segments,
        class_names=class_names,
        seed=42,
        max_train_windows_per_class_for_model=max_train_windows_per_class_for_model,
    )
    score_fn = bitcoinsta_score_fn(model, class_to_idx, bitcoinsta_class)
    bitcoinsta_idx = class_to_idx[bitcoinsta_class]
    print("  Model trained.")

    print("\n[STAGE 4/5] Selecting tau using train-only simulated scenarios...")
    tau_table = select_tau_on_train_scenarios(
        model=model,
        bitcoinsta_idx=bitcoinsta_idx,
        segments_pool=train_segments,
        class_names=class_names,
        bitcoinsta_class=bitcoinsta_class,
        shared_bssid=shared_bssid,
        N_values=N_values,
        mixing_levels=mixing_levels,
        bitcoin_users_per_scenario=bitcoin_users_per_scenario,
        user_agg_num_blocks=user_agg_num_blocks,
        window_duration=window_duration,
        mix_time_jitter_s=mix_time_jitter_s,
        bitcoin_packet_fraction_in_block=bitcoin_packet_fraction_in_block,
        btc_target_packets_per_block=btc_target_packets_per_block,
        min_component_packets=min_component_packets,
        num_val_scenarios_per_setting=num_val_scenarios_per_setting,
        tau_grid_size=tau_grid_size,
        seed=seed_start,
    )

    print("\n[STAGE 4b/5] Calibrating alpha for top-m decision rule (train-only scenarios)...")
    # Calibrate alpha per (N,k) to stabilize thresholding across train/test score scale shifts.
    alpha_table: Dict[Tuple[int, int], float] = {}
    for N in N_values:
        for k in mixing_levels:
            eff_min_m = min_m_pred if int(N) >= int(min_m_pred_apply_N_ge) else 1
            # Gather scenario scores for this setting.
            all_scores = []
            all_labels = []
            scenario_scores: List[np.ndarray] = []
            scenario_labels: List[np.ndarray] = []
            for r in range(num_val_scenarios_per_setting):
                run_seed = seed_start * 100000 + N * 1000 + k * 100 + r + 777
                user_blocks, labels = generate_user_blocks(
                    N=N,
                    mixing_level=k,
                    bitcoin_users_per_scenario=bitcoin_users_per_scenario,
                    segments_pool=train_segments,
                    class_names=class_names,
                    bitcoinsta_class=bitcoinsta_class,
                    shared_bssid=shared_bssid,
                    user_agg_num_blocks=user_agg_num_blocks,
                    window_duration=window_duration,
                    mix_time_jitter_s=mix_time_jitter_s,
                    bitcoin_packet_fraction_in_block=bitcoin_packet_fraction_in_block,
                    btc_target_packets_per_block=btc_target_packets_per_block,
                    min_component_packets=min_component_packets,
                    seed=run_seed,
                )
                scores = compute_user_scores_from_blocks_batched(
                    user_blocks, model=model, bitcoinsta_idx=bitcoinsta_idx
                )
                scenario_scores.append(scores)
                scenario_labels.append(labels)

            # Candidate alphas from [1/N, 0.5]
            alphas = np.linspace(1.0 / float(N), 0.5, int(alpha_grid_size))
            best_alpha = float(alphas[0])
            best_obj = -1.0
            for alpha in alphas:
                m_pred = int(max(eff_min_m, round(alpha * N)))
                f1s = []
                for sc, lb in zip(scenario_scores, scenario_labels):
                    m = compute_metrics_one_scenario_topm(sc, lb, m_pred)["f1"]
                    f1s.append(m)
                # Robust objective: maximize lower-quantile F1 to avoid brittle choices (prevents all-zero in test).
                obj = float(np.quantile(np.array(f1s, dtype=float), 0.2)) if f1s else 0.0
                if obj > best_obj:
                    best_obj = obj
                    best_alpha = float(alpha)

            alpha_table[(int(N), int(k))] = best_alpha
            print(f"  Calibrated alpha for N={N}, k={k}: alpha={best_alpha:.4f} (F1 q20={best_obj:.4f})")

    print("\n[STAGE 5/5] Running held-out attacker-centric evaluation...")
    rows = []
    total_jobs = len(seeds) * len(N_values) * len(mixing_levels)
    done = 0
    for seed in seeds:
        for N in N_values:
            for k in mixing_levels:
                run_seed = seed * 100000 + N * 1000 + k * 10 + 7
                user_blocks, labels = generate_user_blocks(
                    N=N,
                    mixing_level=k,
                    bitcoin_users_per_scenario=bitcoin_users_per_scenario,
                    segments_pool=test_segments,
                    class_names=class_names,
                    bitcoinsta_class=bitcoinsta_class,
                    shared_bssid=shared_bssid,
                    user_agg_num_blocks=user_agg_num_blocks,
                    window_duration=window_duration,
                    mix_time_jitter_s=mix_time_jitter_s,
                    bitcoin_packet_fraction_in_block=bitcoin_packet_fraction_in_block,
                    btc_target_packets_per_block=btc_target_packets_per_block,
                    min_component_packets=min_component_packets,
                    seed=run_seed,
                )
                scores = compute_user_scores_from_blocks_batched(
                    user_blocks, model=model, bitcoinsta_idx=bitcoinsta_idx
                )
                tau = float(tau_table[(int(N), int(k))])
                # Store tau (still useful for reproducibility/ablation)
                rows.append(
                    {
                        "seed": int(seed),
                        "N": int(N),
                        "MixingLevel": int(k),
                        "metric_name": "tau",
                        "value": float(tau),
                    }
                )
                alpha = float(alpha_table[(int(N), int(k))])
                eff_min_m = min_m_pred if int(N) >= int(min_m_pred_apply_N_ge) else 1
                m_pred = int(max(eff_min_m, round(alpha * int(N))))
                rows.append(
                    {
                        "seed": int(seed),
                        "N": int(N),
                        "MixingLevel": int(k),
                        "metric_name": "alpha",
                        "value": float(alpha),
                    }
                )
                rows.append(
                    {
                        "seed": int(seed),
                        "N": int(N),
                        "MixingLevel": int(k),
                        "metric_name": "m_pred",
                        "value": float(m_pred),
                    }
                )

                # Use top-m as the attacker decision rule for binary metrics.
                metrics = compute_metrics_one_scenario_topm(scores, labels, m_pred)

                for metric_name, value in metrics.items():
                    rows.append(
                        {
                            "seed": int(seed),
                            "N": int(N),
                            "MixingLevel": int(k),
                            "metric_name": metric_name,
                            "value": float(value),
                        }
                    )

                done += 1
                if done % max(1, total_jobs // 20) == 0 or done == total_jobs:
                    pct = int(round(100.0 * done / total_jobs))
                    print(f"  Progress: {pct}% ({done}/{total_jobs})")

    df = pd.DataFrame(rows)
    metrics_csv = results_dir / "metrics.csv"
    df.to_csv(metrics_csv, index=False)
    print(f"\nSaved metrics to: {metrics_csv}")

    summary = summarize_metrics(df, metric_names=[])
    make_plots(summary, fig_dir)
    print(f"Saved figures to: {fig_dir}")

    report_path = results_dir / "report.md"
    figures_rel = [os.path.relpath(fig_dir / "attacker_metrics_vs_num_background_apps.png", start=results_dir)]
    write_report(report_path=report_path, config=config, summary_df=summary, figures_rel_paths=figures_rel)
    print(f"Saved report to: {report_path}")


if __name__ == "__main__":
    main()

