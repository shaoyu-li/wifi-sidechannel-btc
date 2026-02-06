# Off-Path Bitcoin Hijacking via Wi-Fi MAC-Layer Side Channels

This repository contains the **full research codebase** accompanying the paper:

> **Off-Path Bitcoin Hijacking via Wi-Fi MAC-Layer Side Channels**

The repository integrates **three complementary research components**:

1. **Wi-Fi–based off-path attack pipeline against Bitcoin nodes**
2. **Machine-learning–based Bitcoin traffic identification**
3. **Large-scale Bitcoin network measurement and probing tools**

Together, these components support the paper’s end-to-end threat model, empirical evaluation, and measurement-driven analysis, as expected for top-tier security venues (e.g., USENIX Security, NDSS, CCS).

---

## Repository Overview

At a high level, the codebase is organized into three logical modules:

- **Attack_Steps/** — End-to-end off-path attack implementation  
- **Bitcoin_Identify_ML/** — Bitcoin traffic detection using ML  
- **Bitcoin_Measurement/** — Bitcoin network measurement and probing  

Each module is self-contained and includes its own README with detailed usage instructions.

---

## 1. Off-Path Attack Pipeline (`Attack_Steps/`)

This module implements the **core contribution** of the paper: a practical off-path attack that reconstructs the TCP connection state of a Bitcoin node by combining Internet-side probing with Wi-Fi MAC-layer side channels.

### Key Capabilities
- Passive identification of Bitcoin traffic from encrypted Wi-Fi frames
- TCP client port inference using Challenge ACK–triggered Wi-Fi signatures
- TCP sequence and ACK number inference via oracle-guided probing
- Protocol-level Bitcoin message injection (`inv`, `tx`) without MITM

### Main Files
- `main.py` — Top-level attack orchestrator
- `InferPort.py` — TCP client port inference
- `InferSeq.py` — TCP sequence inference
- `InferAck.py` — TCP ACK inference
- `Bitcoin_attack/Attack.py` — Bitcoin P2P message injection logic
- `config_attack_step.json` — Attack configuration
- `example_result.json` — Example output

This pipeline operates under a **strict off-path threat model** and does not require Wi-Fi decryption, AP compromise, or traffic redirection.

---

## 2. Bitcoin Traffic Identification (`Bitcoin_Identify_ML/`)

This module implements a **machine-learning–based Bitcoin traffic detector** using features extracted from encrypted Wi-Fi MAC-layer frames.

### Key Capabilities
- Feature extraction from frame length, direction, and timing
- Supervised ML models for Bitcoin traffic classification
- Multi-class classification against background applications
- Pre-trained models for reproducibility

### Main Files
- `bitcoin_detector.py` — ML-based traffic classifier
- `bitcoin_detector_model.pkl` — Trained Bitcoin detection model
- `wifi_multiclass_model.pkl` — Multi-class traffic model
- `data/` — Training and evaluation datasets
- `experiment/` — ML experiments and scripts
- `results/` — Evaluation outputs

This module supports the **victim discovery phase** of the attack and provides empirical evidence that Bitcoin traffic is distinguishable even under encrypted Wi-Fi.

---

## 3. Bitcoin Network Measurement (`Bitcoin_Measurement/`)

This module provides **measurement and probing tools** used to characterize the Bitcoin network and support attack feasibility analysis.

### Key Capabilities
- Discovery of reachable Bitcoin nodes
- Active probing of Bitcoin peers
- Measurement of protocol behavior and responsiveness
- Support for candidate peer set reduction

### Main Files
- `FindReachableNode.py` — Publicly reachable node discovery
- `AckProbing.py` — TCP probing utilities
- `config.yaml` — Measurement configuration

The measurement results inform realistic parameter choices and validate assumptions made in the attack design.

---

## Threat Model Summary

- Attacker is **local but non-privileged** on the victim’s Wi-Fi network
- Attacker can:
  - Passively monitor encrypted IEEE 802.11 frames
  - Send spoofed TCP packets from the Internet
- Attacker cannot:
  - Decrypt Wi-Fi traffic
  - Compromise the AP or victim device
  - Perform MITM or ARP spoofing
- Victim is a **Bitcoin full node behind a NAT-enabled Wi-Fi AP**

---

## Reproducibility and Evaluation

- All experiments are scriptable and configuration-driven
- Noise from background traffic is explicitly considered
- ML models and attack parameters are included for repeatability
- Example configurations and outputs are provided

Refer to the README files inside each subdirectory for module-specific instructions.

---

## Ethics and Responsible Use

This codebase is released **strictly for academic research and defensive analysis**.

Running these tools on networks or systems without explicit authorization may be illegal.  
Users are responsible for complying with all applicable laws and institutional policies.



---

## Contact

For questions regarding experimental setup or reproducibility, please contact the authors via the paper’s official channels.
