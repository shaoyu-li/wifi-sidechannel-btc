
from __future__ import annotations

import argparse
import csv
import ipaddress
import logging
import os
import socket
import struct
import sys
import time
from typing import Iterable, List, Optional, Tuple


def _detect_dialect(sample: str) -> csv.Dialect:
    try:
        return csv.Sniffer().sniff(sample)
    except Exception:
        return csv.excel


def read_addresses(path: str) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with open(path, "r", encoding="utf-8", newline="") as f:
        sample = f.read(2048)
        f.seek(0)
        dialect = _detect_dialect(sample)
        reader = csv.reader(f, dialect)

        addresses: List[str] = []

        for row in reader:
            if not row:
                continue
            value = row[0].strip()
            
            if len(row) == 2 and row[1].strip().isdigit():
                value = f"{row[0].strip()}:{row[1].strip()}"
            if not value or value.startswith("#"):
                continue
            addresses.append(value)

        return addresses


def _write_csv_lines(lines: Iterable[str], out_path: str) -> None:
    if out_path == "-" or out_path is None:
        writer = csv.writer(sys.stdout)
        for line in lines:
            writer.writerow([line])
        return

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for line in lines:
            writer.writerow([line])


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Filter IPv4 address:port entries to those reachable via Bitcoin P2P handshake."
        )
    )
    p.add_argument(
        "-i",
        "--input",
        default="ipv4_nodes.csv",
        help="Input CSV file containing IPv4 address:port entries.",
    )
    p.add_argument(
        "-o",
        "--out",
        default="-",
        help="Output CSV path for reachable addresses or '-' for stdout.",
    )
    p.add_argument(
        "--not-out",
        default="not_reachable.csv",
        help="CSV path for addresses not reachable (default: not_reachable.csv).",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="Per-node connect/handshake timeout in seconds.",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging.")
    return p


def _is_ipv4_host(host: str) -> bool:
    try:
        ipaddress.IPv4Address(host)
        return True
    except Exception:
        return False


def _split_host_port(addr: str) -> Optional[Tuple[str, int]]:
    if addr.count(":") != 1:
        return None
    host, port_str = addr.rsplit(":", 1)
    host = host.strip()
    try:
        port = int(port_str)
    except Exception:
        return None
    if port < 1 or port > 65535:
        return None
    if not _is_ipv4_host(host):
        return None
    return host, port


def _sha256(data: bytes) -> bytes:
    import hashlib

    return hashlib.sha256(data).digest()


def _checksum(payload: bytes) -> bytes:
    return _sha256(_sha256(payload))[:4]


def _build_version_payload() -> bytes:
    version = 70016
    services = 0
    timestamp = int(time.time())
    addr_services = 0
    addr_ip = b"\x00" * 16
    addr_port = 0
    nonce = int.from_bytes(os.urandom(8), "little", signed=False)
    user_agent = b"/reachable_from_list:0.1/"
    start_height = 0
    relay = 0

    payload = b""
    payload += struct.pack("<iQQ", version, services, timestamp)
    payload += struct.pack("<Q", addr_services)
    payload += addr_ip
    payload += struct.pack(">H", addr_port)
    payload += struct.pack("<Q", addr_services)
    payload += addr_ip
    payload += struct.pack(">H", addr_port)
    payload += struct.pack("<Q", nonce)
    payload += struct.pack("B", len(user_agent)) + user_agent
    payload += struct.pack("<i", start_height)
    payload += struct.pack("<?", relay)
    return payload


def _build_message(command: bytes, payload: bytes) -> bytes:
    magic = 0xD9B4BEF9
    command_padded = command.ljust(12, b"\x00")
    length = len(payload)
    return struct.pack("<L12sL4s", magic, command_padded, length, _checksum(payload)) + payload


def _recv_message(sock: socket.socket, timeout: float) -> Optional[Tuple[str, bytes]]:
    sock.settimeout(timeout)
    header = b""
    while len(header) < 24:
        chunk = sock.recv(24 - len(header))
        if not chunk:
            return None
        header += chunk
    magic, cmd, length, checksum = struct.unpack("<L12sL4s", header)
    if magic != 0xD9B4BEF9:
        return None
    cmd = cmd.rstrip(b"\x00").decode("ascii", errors="ignore")
    payload = b""
    while len(payload) < length:
        chunk = sock.recv(length - len(payload))
        if not chunk:
            return None
        payload += chunk
    if _checksum(payload) != checksum:
        return None
    return cmd, payload


def _probe_p2p(host: str, port: int, timeout: float) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout) as sock:
            sock.settimeout(timeout)
            version_payload = _build_version_payload()
            sock.sendall(_build_message(b"version", version_payload))
            got_version = False
            got_verack = False
            start = time.time()
            while time.time() - start < timeout:
                msg = _recv_message(sock, timeout)
                if msg is None:
                    return False
                cmd, _payload = msg
                if cmd == "version":
                    got_version = True
                    sock.sendall(_build_message(b"verack", b""))
                elif cmd == "verack":
                    got_verack = True
                if got_version and got_verack:
                    return True
    except Exception:
        return False
    return False


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    try:
        addresses = read_addresses(args.input)
    except Exception as e:
        logging.error("Failed to read input: %s", e)
        return 2

    if not addresses:
        logging.warning("No addresses found in input.")
        return 0

    reachable: List[str] = []
    not_reachable: List[str] = []
    skipped: List[str] = []

    for addr in addresses:
        parsed = _split_host_port(addr)
        if parsed is None:
            skipped.append(addr)
            continue
        host, port = parsed
        ok = _probe_p2p(host, port, args.timeout)
        if ok:
            reachable.append(addr)
        else:
            not_reachable.append(addr)

    _write_csv_lines(reachable, args.out)
    if args.not_out:
        _write_csv_lines(not_reachable, args.not_out)

    logging.info(
        "Input: %d | reachable: %d | not reachable: %d | skipped: %d",
        len(addresses),
        len(reachable),
        len(not_reachable),
        len(skipped),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
