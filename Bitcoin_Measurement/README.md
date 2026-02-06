# Bitcoin_Measurement

This module implements the Internet-side Bitcoin measurement component used in our paper.
It consists of two scripts that (i) identify publicly reachable Bitcoin peers and (ii) perform
TCP-level challenge-ACK probing on established Bitcoin connections.

The code is designed for **measurement and reproducibility**, not for exploitation.
All experiments should be conducted in controlled environments and in accordance with
ethical research guidelines.

---

## Overview

This module contains two main scripts:

- **`FindReachableNode.py`**  
  Scans a list of candidate Bitcoin nodes and identifies those that are *publicly reachable*
  by completing a valid Bitcoin P2P handshake (`version` / `verack`).

- **`AckProbing.py`**  
  Establishes a real TCP connection to a reachable Bitcoin peer, completes the Bitcoin handshake,
  and injects out-of-window TCP ACK probes to measure challenge-ACK style responses from the peer.

Together, these scripts support the measurement pipeline described in the paper:

1. Reduce the global node list to a set of reachable peers.
2. Perform targeted TCP-level probing only against validated peers.

---

## Requirements

- Linux (Ubuntu 22.04 recommended)
- Python ≥ 3.8
- Root privilege is required **only for `AckProbing.py`**
  (raw packet injection and packet sniffing).

### Python dependencies
```bash
pip install scapy pyyaml

