# Attack_Step Pipeline

This project implements an end-to-end off-path Bitcoin attack pipeline based on
Wi-Fi side-channel observations and TCP state inference.

The pipeline performs:

1. TCP client port inference (InferPort.py)
2. TCP sequence number inference (InferSeq.py)
3. TCP ACK number inference (InferAck.py)
4. Execution of `Bitcoin_attack/Attack.py` using the inferred TCP state

`main.py` does NOT perform TCP injection or TCP DoS itself.
It only orchestrates inference and then executes `Bitcoin_attack/Attack.py`.

---

## Directory Layout

```
project_root/
├── main.py
├── InferPort.py
├── InferSeq.py
├── InferAck.py
├── config_attack_step.json
├── README.md
└── Bitcoin_attack/
    └── Attack.py
```

---

## Requirements

- Linux or macOS
- Python ≥ 3.8
- Scapy
- Root privileges (raw packets + sniffing)

Install dependency:
```bash
pip install scapy
```

---

## Usage

### Full pipeline execution
```bash
sudo python3 main.py --config config_attack_step.json
```

### Dry run (no packets sent)
```bash
sudo python3 main.py --config config_attack_step.json --dry-run
```

### Save inference results
```bash
sudo python3 main.py --config config_attack_step.json --save result.json
```

---

## Pipeline Description

### Step 1 – Client Port Inference
- Sends out-of-window ACK probes
- Detects Challenge ACK via encrypted Wi-Fi frame length
- Output: `client_port`

### Step 2 – TCP Sequence Inference
- Binary probing of TCP sequence space
- Uses Wi-Fi side-channel signature
- Output: `seq_next`

### Step 3 – TCP ACK Inference
- Binary probing of ACK-related window boundary
- Output: `ack_value`

### Step 4 – Bitcoin Protocol Attack
- `main.py` executes `Bitcoin_attack/Attack.py` as-is
- Automatically supplies inferred `seq` and `ack`
- Injects `inv`, waits for `getdata`, then injects `tx`

---

## Important Notes

- `expected_raw_len` must match your Wi-Fi frame signature exactly
- `sniff_if_name` must be monitor-mode interfaces
- Attack 4-tuple (src/dst IP & port) is explicitly defined in config
- `dst_port` in `step4_attack_py` should be the inferred client port

---

## Legal Notice

This code is for academic research and controlled experiments only.
Running it on networks or systems without authorization may be illegal.
