"""WiFi traffic classifier + Bitcoin-in-mixture evaluation (per-file classes).

Each JSON file in `data/` is treated as its own class (multi-class training on pure windows).
Evaluation mixes increasing numbers of distinct classes in held-out test windows to test:
1) Bitcoin presence detection (bitcoinsta present vs absent)
2) Bitcoin user attribution under synthetic multi-user mixing (MAC remapping)
"""

import json
import math
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    import numpy as np
    import pandas as pd
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import train_test_split
    import xgboost as xgb
    import joblib
except ImportError as e:
    print(f"Error importing required libraries: {e}")
    print("\nPlease install required packages:")
    print("  pip install numpy pandas scikit-learn xgboost joblib")
    print("\nOr if using conda:")
    print("  conda install numpy pandas scikit-learn xgboost joblib")
    sys.exit(1)


def load_json_file(path: Path) -> List[Dict]:
    try:
        with open(path, "r") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("JSON root is not a list")
        return data
    except (json.JSONDecodeError, MemoryError, OSError, ValueError) as e:
        print(f"Warning: Failed to load {path}: {e}")
        return []


def load_data_classes(data_root: str) -> Dict[str, List[Dict]]:
    """Load flat `data/*.json`, treating each filename as a class."""
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    out: Dict[str, List[Dict]] = {}
    for p in sorted(root.glob("*.json")):
        class_name = p.stem
        packets = load_json_file(p)
        if len(packets) == 0:
            print(f"  Skipping empty/unreadable class file: {p.name}")
            continue
        out[class_name] = packets
    return out


def segment_packets_by_time_window(
    packets: List[Dict],
    window_duration: float = 0.2,
    step_size: float = 0.1,
    min_frames: int = 10
) -> List[List[Dict]]:
    """Segment packets into sliding time windows (200ms / 100ms step by default)."""
    if len(packets) == 0:
        return []

    sorted_packets = sorted(packets, key=lambda x: x.get('ts', 0))
    if len(sorted_packets) == 0:
        return []

    start_time = sorted_packets[0].get("ts", 0)
    end_time = sorted_packets[-1].get("ts", 0)
    if end_time <= start_time:
        return []

    windows: List[List[Dict]] = []
    current_start = start_time
    left = 0
    right = 0
    n = len(sorted_packets)

    # Two-pointer windowing: O(n + number_of_windows)
    while current_start < end_time:
        window_end = current_start + window_duration

        while left < n and sorted_packets[left].get("ts", 0) < current_start:
            left += 1
        if right < left:
            right = left
        while right < n and sorted_packets[right].get("ts", 0) < window_end:
            right += 1

        if right - left >= min_frames:
            windows.append(sorted_packets[left:right])

        current_start += step_size

    return windows


def extract_features(packets: List[Dict]) -> np.ndarray:
    """Extract statistical features from packet sequence.
    
    Args:
        packets: List of packet dictionaries
        
    Returns:
        Feature vector as numpy array (50 features)
    """
    if len(packets) == 0:
        return np.zeros(50)
    
    df = pd.DataFrame(packets)
    
    features = []
    
    # Packet size features (3 columns * 5 stats = 15 features)
    for col in ['phy_len', 'mac_len', 'enc_payload_len']:
        if col in df.columns:
            features.extend([
                df[col].mean(),
                df[col].std() if len(df) > 1 else 0.0,
                df[col].min(),
                df[col].max(),
                df[col].median()
            ])
        else:
            features.extend([0.0] * 5)
    
    # Timing features (3 features)
    if 'ts' in df.columns and len(df) > 1:
        df_sorted = df.sort_values('ts')
        intervals = df_sorted['ts'].diff().dropna()
        if len(intervals) > 0:
            features.append(intervals.mean())
            features.append(intervals.std())
        else:
            features.extend([0.0, 0.0])
        
        duration = df['ts'].max() - df['ts'].min()
        packet_rate = len(df) / duration if duration > 0 else 0.0
        features.append(packet_rate)
    else:
        features.extend([0.0, 0.0, 0.0])
    
    # Direction features (2 features)
    if 'direction' in df.columns:
        tx_count = (df['direction'] == 'tx').sum()
        rx_count = (df['direction'] == 'rx').sum()
        total = tx_count + rx_count
        tx_ratio = tx_count / total if total > 0 else 0.0
        rx_ratio = rx_count / total if total > 0 else 0.0
        features.extend([tx_ratio, rx_ratio])
    else:
        features.extend([0.0, 0.0])
    
    # Type distribution (4 features)
    if 'type' in df.columns:
        type_counts = df['type'].value_counts()
        for t in [0, 1, 2, 3]:
            count = type_counts.get(t, 0)
            features.append(count / len(df) if len(df) > 0 else 0.0)
    else:
        features.extend([0.0] * 4)
    
    # Subtype distribution (5 features - top 5 subtypes)
    if 'subtype' in df.columns:
        subtype_counts = df['subtype'].value_counts()
        top_subtypes = sorted(subtype_counts.head(5).index)
        subtype_features = []
        for st in top_subtypes[:5]:
            count = subtype_counts.get(st, 0)
            subtype_features.append(count / len(df) if len(df) > 0 else 0.0)
        # Pad to exactly 5 features
        while len(subtype_features) < 5:
            subtype_features.append(0.0)
        features.extend(subtype_features[:5])
    else:
        features.extend([0.0] * 5)
    
    # RSSI features (2 features)
    if 'rssi' in df.columns:
        features.append(df['rssi'].mean())
        features.append(df['rssi'].std() if len(df) > 1 else 0.0)
    else:
        features.extend([0.0, 0.0])
    
    # MAC address features (3 features)
    for col in ['sa', 'da', 'bssid']:
        if col in df.columns:
            unique_count = df[col].nunique()
            features.append(unique_count)
        else:
            features.append(0.0)
    
    # Protected packet ratio (1 feature)
    if 'protected' in df.columns:
        protected_ratio = (df['protected'] == 1).sum() / len(df) if len(df) > 0 else 0.0
        features.append(protected_ratio)
    else:
        features.append(0.0)
    
    # Retry ratio (1 feature)
    if 'retry' in df.columns:
        retry_ratio = (df['retry'] == 1).sum() / len(df) if len(df) > 0 else 0.0
        features.append(retry_ratio)
    else:
        features.append(0.0)
    
    # Additional features to reach 50 total
    # Packet count
    features.append(len(df))
    
    # Size ratio features
    if 'phy_len' in df.columns and 'enc_payload_len' in df.columns:
        size_ratio = (df['enc_payload_len'] / (df['phy_len'] + 1e-10)).mean()
        features.append(size_ratio)
    else:
        features.append(0.0)
    
    # Packet size percentiles
    if 'phy_len' in df.columns:
        features.append(df['phy_len'].quantile(0.25))
        features.append(df['phy_len'].quantile(0.75))
    else:
        features.extend([0.0, 0.0])
    
    # Direction change frequency
    if 'direction' in df.columns and len(df) > 1:
        direction_changes = (df['direction'].shift(1) != df['direction']).sum()
        features.append(direction_changes / len(df))
    else:
        features.append(0.0)
    
    # RSSI percentiles
    if 'rssi' in df.columns:
        features.append(df['rssi'].min())
        features.append(df['rssi'].max())
        features.append(df['rssi'].median())
    else:
        features.extend([0.0, 0.0, 0.0])
    
    # MAC address diversity (ratio of unique to total)
    if 'sa' in df.columns:
        sa_diversity = df['sa'].nunique() / len(df) if len(df) > 0 else 0.0
        features.append(sa_diversity)
    else:
        features.append(0.0)
    
    if 'da' in df.columns:
        da_diversity = df['da'].nunique() / len(df) if len(df) > 0 else 0.0
        features.append(da_diversity)
    else:
        features.append(0.0)
    
    # BSSID diversity
    if 'bssid' in df.columns:
        bssid_diversity = df['bssid'].nunique() / len(df) if len(df) > 0 else 0.0
        features.append(bssid_diversity)
    else:
        features.append(0.0)
    
    # Encrypted payload size statistics
    if 'enc_payload_len' in df.columns:
        features.append(df['enc_payload_len'].quantile(0.25))
        features.append(df['enc_payload_len'].quantile(0.75))
        # Coefficient of variation for packet sizes
        cv = df['enc_payload_len'].std() / (df['enc_payload_len'].mean() + 1e-10)
        features.append(cv)
    else:
        features.extend([0.0, 0.0, 0.0])
    
    # Ensure exactly 50 features
    features = features[:50]
    while len(features) < 50:
        features.append(0.0)
    
    return np.array(features)


def time_split_packets(packets: List[Dict], train_frac: float) -> Tuple[List[Dict], List[Dict]]:
    """Time-based split within a single trace to prevent window leakage."""
    if len(packets) == 0:
        return [], []
    sorted_packets = sorted(packets, key=lambda x: x.get("ts", 0))
    ts0 = sorted_packets[0].get("ts", 0)
    ts1 = sorted_packets[-1].get("ts", 0)
    if ts1 <= ts0:
        mid = int(len(sorted_packets) * train_frac)
        return sorted_packets[:mid], sorted_packets[mid:]
    cutoff = ts0 + train_frac * (ts1 - ts0)
    train = [p for p in sorted_packets if p.get("ts", 0) < cutoff]
    test = [p for p in sorted_packets if p.get("ts", 0) >= cutoff]
    return train, test


def infer_shared_bssid(packets: List[Dict], max_packets: int = 20000) -> Optional[str]:
    vals = []
    for p in packets[:max_packets]:
        b = p.get("bssid")
        if isinstance(b, str) and len(b) >= 11:
            vals.append(b)
    if not vals:
        return None
    return Counter(vals).most_common(1)[0][0]


def collect_windows_limited(
    packets: List[Dict],
    window_duration: float,
    step_size: float,
    min_frames: int,
    max_windows: int,
) -> List[List[Dict]]:
    """Collect up to `max_windows` windows (early-stop for speed)."""
    if max_windows <= 0:
        return []
    windows: List[List[Dict]] = []
    # Use the same segmentation algorithm but early-stop.
    sorted_packets = sorted(packets, key=lambda x: x.get("ts", 0))
    if not sorted_packets:
        return []
    start_time = sorted_packets[0].get("ts", 0)
    end_time = sorted_packets[-1].get("ts", 0)
    if end_time <= start_time:
        return []

    current_start = start_time
    left = 0
    right = 0
    n = len(sorted_packets)
    while current_start < end_time and len(windows) < max_windows:
        window_end = current_start + window_duration
        while left < n and sorted_packets[left].get("ts", 0) < current_start:
            left += 1
        if right < left:
            right = left
        while right < n and sorted_packets[right].get("ts", 0) < window_end:
            right += 1
        if right - left >= min_frames:
            windows.append(sorted_packets[left:right])
        current_start += step_size
    return windows


def pseudo_station_mac(user_idx: int) -> str:
    """Locally administered unicast MAC."""
    # 02:00:00:00:00:XX
    return f"02:00:00:00:00:{user_idx:02x}"


def remap_component_for_user(
    packets: List[Dict],
    user_idx: int,
    shared_bssid: Optional[str],
    time_jitter_s: float,
    rng: random.Random,
) -> List[Dict]:
    """Remap station MACs to a pseudo-user and normalize time to start at 0 (+ jitter)."""
    if not packets:
        return []

    station = pseudo_station_mac(user_idx)
    t0 = packets[0].get("ts", 0)
    jitter = rng.uniform(0.0, time_jitter_s) if time_jitter_s > 0 else 0.0

    out: List[Dict] = []
    for p in packets:
        q = dict(p)  # shallow copy
        ts = q.get("ts", 0)
        q["ts"] = (ts - t0) + jitter

        if shared_bssid is not None:
            q["bssid"] = shared_bssid

        for addr in ("sa", "da"):
            v = q.get(addr)
            if isinstance(v, str) and shared_bssid is not None and v != shared_bssid:
                q[addr] = station
        out.append(q)

    out.sort(key=lambda x: x.get("ts", 0))
    return out


def packet_user_idx(p: Dict, station_mac_to_user: Dict[str, int]) -> Optional[int]:
    sa = p.get("sa")
    da = p.get("da")
    if sa in station_mac_to_user:
        return station_mac_to_user[sa]
    if da in station_mac_to_user:
        return station_mac_to_user[da]
    return None


@dataclass(frozen=True)
class MixedSample:
    packets: List[Dict]
    has_bitcoinsta: int  # 1/0
    true_bitcoin_user: Optional[int]  # only for positives
    station_mac_to_user: Dict[str, int]


def build_train_test_segments_from_classes(
    classes_packets: Dict[str, List[Dict]],
    *,
    train_frac: float,
    window_duration: float,
    step_size: float,
    min_frames: int,
    max_windows_train_per_class: int,
    max_windows_test_per_class: int,
) -> Tuple[Dict[str, List[List[Dict]]], Dict[str, List[List[Dict]]]]:
    """Leakage-safe per-class time split + windowing for reuse in experiments."""
    class_names = sorted(classes_packets.keys())
    train_segments: Dict[str, List[List[Dict]]] = {}
    test_segments: Dict[str, List[List[Dict]]] = {}
    for cn in class_names:
        packets = classes_packets[cn]
        tr_pkts, te_pkts = time_split_packets(packets, train_frac=train_frac)
        train_segments[cn] = collect_windows_limited(
            tr_pkts,
            window_duration=window_duration,
            step_size=step_size,
            min_frames=min_frames,
            max_windows=max_windows_train_per_class,
        )
        test_segments[cn] = collect_windows_limited(
            te_pkts,
            window_duration=window_duration,
            step_size=step_size,
            min_frames=min_frames,
            max_windows=max_windows_test_per_class,
        )
    return train_segments, test_segments


def train_multiclass_xgb_from_segments(
    train_segments: Dict[str, List[List[Dict]]],
    *,
    class_names: List[str],
    seed: int,
    max_train_windows_per_class_for_model: int,
    xgb_params: Optional[Dict] = None,
) -> Tuple["xgb.XGBClassifier", Dict[str, int]]:
    """Train multi-class XGBoost on pure windows from the training split."""
    rng = random.Random(seed)
    class_to_idx = {cn: i for i, cn in enumerate(class_names)}

    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    for cn in class_names:
        segs = train_segments.get(cn, [])
        if not segs:
            continue
        take = min(len(segs), max_train_windows_per_class_for_model)
        idxs = list(range(len(segs)))
        rng.shuffle(idxs)
        for si in idxs[:take]:
            X_list.append(extract_features(segs[si]))
            y_list.append(class_to_idx[cn])

    X = np.vstack(X_list) if X_list else np.zeros((0, 50), dtype=float)
    y = np.array(y_list, dtype=int)
    if X.shape[0] == 0:
        raise RuntimeError("No training samples were constructed.")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    params = dict(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=seed,
        objective="multi:softprob",
        num_class=len(class_names),
        eval_metric="mlogloss",
        n_jobs=0,
    )
    if xgb_params:
        params.update(xgb_params)

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model, class_to_idx


def bitcoinsta_score_fn(
    model: "xgb.XGBClassifier",
    class_to_idx: Dict[str, int],
    bitcoinsta_class: str,
):
    """Return a function packets->p(bitcoinsta_class)."""
    if bitcoinsta_class not in class_to_idx:
        raise KeyError(f"bitcoinsta_class `{bitcoinsta_class}` not in class_to_idx")
    idx = class_to_idx[bitcoinsta_class]

    def _score(packets_window: List[Dict]) -> float:
        feats = extract_features(packets_window).reshape(1, -1)
        proba = model.predict_proba(feats)[0]
        return float(proba[idx])

    return _score


def main():
    """Main execution function."""
    # -------------------------
    # Experiment configuration
    # -------------------------
    seed = 42
    rng = random.Random(seed)

    data_root = "data"
    bitcoinsta_class = "bitcoinsta"

    # Sliding window parameters
    window_duration = 0.2
    step_size = 0.1
    min_frames = 10

    # Speed knobs (keep things directly runnable on large traces)
    train_frac = 0.7
    max_windows_train_per_class = 5000
    max_windows_test_per_class = 5000
    max_train_windows_per_class_for_model = 5000

    # Mixture evaluation knobs
    k_values = [1, 2, 3, 4, 5]
    n_test_pos_per_k = 400
    n_test_neg_per_k = 400
    n_val_mix_per_k = 200  # used only for tau selection (train-split only)
    mix_time_jitter_s = 0.02

    print("=" * 72)
    print("WIFI TRAFFIC CLASSIFIER + BITCOIN MIXTURE EVALUATION (per-file classes)")
    print("=" * 72)

    print("\n[STEP 1/9] Loading per-file classes from `data/*.json`...")
    classes_packets = load_data_classes(data_root)
    class_names = sorted(classes_packets.keys())
    print(f"  Loaded {len(class_names)} classes: {class_names}")
    if bitcoinsta_class not in classes_packets:
        raise RuntimeError(f"Required Bitcoin class file not found: {bitcoinsta_class}.json in {data_root}/")

    # Shared BSSID for synthetic multi-user mixing (keep constant across users)
    shared_bssid = infer_shared_bssid(classes_packets[bitcoinsta_class])
    print(f"  Inferred shared BSSID: {shared_bssid}")

    print("\n[STEP 2/9] Time-splitting each class into train/test and collecting windows...")
    train_segments: Dict[str, List[List[Dict]]] = {}
    test_segments: Dict[str, List[List[Dict]]] = {}
    for i, cn in enumerate(class_names):
        packets = classes_packets[cn]
        tr_pkts, te_pkts = time_split_packets(packets, train_frac=train_frac)

        tr_w = collect_windows_limited(
            tr_pkts, window_duration=window_duration, step_size=step_size,
            min_frames=min_frames, max_windows=max_windows_train_per_class
        )
        te_w = collect_windows_limited(
            te_pkts, window_duration=window_duration, step_size=step_size,
            min_frames=min_frames, max_windows=max_windows_test_per_class
        )
        train_segments[cn] = tr_w
        test_segments[cn] = te_w
        print(f"  Class {i+1}/{len(class_names)} `{cn}`: train_windows={len(tr_w)}, test_windows={len(te_w)}")

    print("\n[STEP 3/9] Building pure-window training set (multi-class)...")
    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    class_to_idx = {cn: j for j, cn in enumerate(class_names)}

    for cn in class_names:
        segs = train_segments.get(cn, [])
        if not segs:
            print(f"  Warning: class `{cn}` has 0 train windows; skipping in training.")
            continue
        take = min(len(segs), max_train_windows_per_class_for_model)
        # deterministic but shuffled
        idxs = list(range(len(segs)))
        rng.shuffle(idxs)
        idxs = idxs[:take]
        for jj, si in enumerate(idxs):
            feats = extract_features(segs[si])
            X_list.append(feats)
            y_list.append(class_to_idx[cn])
        print(f"  Added {take} windows for class `{cn}`")

    X = np.vstack(X_list) if X_list else np.zeros((0, 50), dtype=float)
    y = np.array(y_list, dtype=int)
    if X.shape[0] == 0:
        raise RuntimeError("No training samples were constructed. Check windowing thresholds.")
    print(f"  Training samples: {X.shape[0]}, features per sample: {X.shape[1]}")

    print("\n[STEP 4/9] Train/validation split + multi-class XGBoost training...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    print(f"  Train: {X_train.shape[0]} samples; Val: {X_val.shape[0]} samples")

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=seed,
        objective="multi:softprob",
        num_class=len(class_names),
        eval_metric="mlogloss",
        n_jobs=0,
    )

    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    print("  Model training completed.")

    print("\n[STEP 5/9] Saving model...")
    joblib.dump(
        {"model": model, "class_names": class_names, "seed": seed},
        "wifi_multiclass_model.pkl",
    )
    print("  Saved: wifi_multiclass_model.pkl")

    bitcoinsta_idx = class_to_idx[bitcoinsta_class]

    def score_bitcoinsta_presence(packets_window: List[Dict]) -> float:
        feats = extract_features(packets_window).reshape(1, -1)
        proba = model.predict_proba(feats)[0]
        return float(proba[bitcoinsta_idx])

    def sample_window_from_pool(pool: List[List[Dict]]) -> List[Dict]:
        if not pool:
            raise RuntimeError("Empty window pool encountered during mixing.")
        return pool[rng.randrange(len(pool))]

    def make_mixed_sample(
        k_other: int,
        use_bitcoinsta: bool,
        segments_pool: Dict[str, List[List[Dict]]],
    ) -> MixedSample:
        non_btc_classes = [c for c in class_names if c != bitcoinsta_class]
        k_other = max(0, min(k_other, len(non_btc_classes)))

        chosen: List[str] = []
        if use_bitcoinsta:
            chosen.append(bitcoinsta_class)
            others = rng.sample(non_btc_classes, k_other) if k_other > 0 else []
            chosen.extend(others)
        else:
            # For negatives, target up to k_other+1 distinct non-bitcoin classes (capped by available).
            target_distinct = min(k_other + 1, len(non_btc_classes))
            chosen.extend(rng.sample(non_btc_classes, target_distinct))

        # Assign each component to a pseudo-user, remap MACs, normalize time, then merge.
        station_mac_to_user: Dict[str, int] = {}
        remapped_components: List[List[Dict]] = []

        true_btc_user: Optional[int] = None
        for user_idx, cn in enumerate(chosen):
            seg = sample_window_from_pool(segments_pool[cn])
            remapped = remap_component_for_user(
                seg, user_idx=user_idx, shared_bssid=shared_bssid,
                time_jitter_s=mix_time_jitter_s, rng=rng
            )
            remapped_components.append(remapped)
            station_mac_to_user[pseudo_station_mac(user_idx)] = user_idx
            if cn == bitcoinsta_class:
                true_btc_user = user_idx

        mixed = [p for comp in remapped_components for p in comp]
        mixed.sort(key=lambda x: x.get("ts", 0))
        return MixedSample(
            packets=mixed,
            has_bitcoinsta=1 if use_bitcoinsta else 0,
            true_bitcoin_user=true_btc_user,
            station_mac_to_user=station_mac_to_user,
        )

    print("\n[STEP 6/9] Selecting threshold tau for bitcoinsta presence (train-split only)...")
    val_scores: List[float] = []
    val_labels: List[int] = []
    for k in k_values:
        for _ in range(n_val_mix_per_k):
            s_pos = make_mixed_sample(k_other=k, use_bitcoinsta=True, segments_pool=train_segments)
            val_scores.append(score_bitcoinsta_presence(s_pos.packets))
            val_labels.append(1)
            s_neg = make_mixed_sample(k_other=k, use_bitcoinsta=False, segments_pool=train_segments)
            val_scores.append(score_bitcoinsta_presence(s_neg.packets))
            val_labels.append(0)
        print(f"  Built validation mixtures for k={k}: {2*n_val_mix_per_k} samples")

    scores = np.array(val_scores, dtype=float)
    labels = np.array(val_labels, dtype=int)
    # Candidate thresholds from score quantiles.
    candidates = np.unique(np.quantile(scores, np.linspace(0.05, 0.95, 41)))
    best_tau = 0.5
    best_f1 = -1.0
    for tau in candidates:
        pred = (scores >= tau).astype(int)
        tp = int(((pred == 1) & (labels == 1)).sum())
        fp = int(((pred == 1) & (labels == 0)).sum())
        fn = int(((pred == 0) & (labels == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_tau = float(tau)
    print(f"  Selected tau={best_tau:.4f} (val F1={best_f1:.4f})")

    print("\n[STEP 7/9] Held-out test evaluation: mixture sweep k=1..5 (bitcoinsta present vs absent)...")
    for k in k_values:
        y_true: List[int] = []
        y_pred: List[int] = []
        attr_correct = 0
        attr_total = 0

        total = n_test_pos_per_k + n_test_neg_per_k
        for idx in range(total):
            is_pos = idx < n_test_pos_per_k
            sample = make_mixed_sample(k_other=k, use_bitcoinsta=is_pos, segments_pool=test_segments)
            score = score_bitcoinsta_presence(sample.packets)
            pred_present = 1 if score >= best_tau else 0
            y_true.append(sample.has_bitcoinsta)
            y_pred.append(pred_present)

            # Attribution only meaningful on positives.
            if sample.has_bitcoinsta == 1 and sample.true_bitcoin_user is not None:
                # Partition packets by pseudo-user and pick argmax p(bitcoinsta)
                user_packets: Dict[int, List[Dict]] = {}
                for p in sample.packets:
                    ui = packet_user_idx(p, sample.station_mac_to_user)
                    if ui is None:
                        continue
                    user_packets.setdefault(ui, []).append(p)

                best_user = None
                best_user_score = -math.inf
                for ui, pkts in user_packets.items():
                    if len(pkts) < min_frames:
                        continue
                    s_ui = score_bitcoinsta_presence(pkts)
                    if s_ui > best_user_score:
                        best_user_score = s_ui
                        best_user = ui
                if best_user is not None:
                    attr_total += 1
                    if best_user == sample.true_bitcoin_user:
                        attr_correct += 1

            if (idx + 1) % max(1, total // 5) == 0:
                pct = int(round(100 * (idx + 1) / total))
                print(f"  Mixture k={k}: {pct}%")

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        acc = (np.array(y_true) == np.array(y_pred)).mean()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        attr_acc = attr_correct / attr_total if attr_total > 0 else 0.0

        print(f"\n  [RESULT k={k}] Bitcoin-present detection on held-out mixtures")
        print(f"    Accuracy: {acc:.4f}  Precision: {precision:.4f}  Recall: {recall:.4f}  F1: {f1:.4f}  FPR: {fpr:.4f}")
        print(f"    Confusion matrix [[TN, FP],[FN, TP]] = {cm.tolist()}")
        print(f"    Bitcoin-user attribution accuracy (positives): {attr_acc:.4f} (n={attr_total})")

    print("\n[STEP 8/9] Quick sanity: multi-class validation on pure windows (optional summary)...")
    y_val_pred = model.predict(X_val)
    print(classification_report(y_val, y_val_pred, digits=4))

    print("\n[STEP 9/9] Done.")
    print("=" * 72)


if __name__ == "__main__":
    main()

