import argparse
import os
import random
import socket
import struct
import sys
import time
from dataclasses import dataclass
from typing import Optional, Tuple, List

# Scapy
try:
    from scapy.all import IP, TCP, sniff, send, conf
    conf.verb = 0
except ImportError:
    print("ERROR: scapy is required. Install with: pip install scapy")
    sys.exit(1)



MAINNET_MAGIC = 0xD9B4BEF9

def _sha256d(b: bytes) -> bytes:
    import hashlib
    return hashlib.sha256(hashlib.sha256(b).digest()).digest()

def _checksum(payload: bytes) -> bytes:
    return _sha256d(payload)[:4]

def _pack_varstr(s: bytes) -> bytes:
    # Bitcoin var_str: <varint len> + bytes
    n = len(s)
    return _pack_varint(n) + s

def _pack_varint(n: int) -> bytes:
    if n < 0xfd:
        return struct.pack("<B", n)
    elif n <= 0xffff:
        return b"\xfd" + struct.pack("<H", n)
    elif n <= 0xffffffff:
        return b"\xfe" + struct.pack("<I", n)
    else:
        return b"\xff" + struct.pack("<Q", n)

def _build_msg(command: str, payload: bytes, magic: int = MAINNET_MAGIC) -> bytes:
    cmd = command.encode("ascii")
    if len(cmd) > 12:
        raise ValueError("command too long")
    cmd_padded = cmd + b"\x00" * (12 - len(cmd))
    length = struct.pack("<I", len(payload))
    cksum = _checksum(payload)
    header = struct.pack("<I", magic) + cmd_padded + length + cksum
    return header + payload

def _recvall(sock: socket.socket, n: int, timeout: float) -> Optional[bytes]:
    sock.settimeout(timeout)
    data = b""
    while len(data) < n:
        try:
            chunk = sock.recv(n - len(data))
        except socket.timeout:
            return None
        if not chunk:
            return None
        data += chunk
    return data

def _recv_msg(sock: socket.socket, timeout: float, max_payload: int = 4 * 1024 * 1024) -> Optional[Tuple[str, bytes]]:
    """
    Receive a Bitcoin P2P message: (command, payload)
    Adds max_payload bound to avoid pathological reads.
    """
    header = _recvall(sock, 24, timeout)
    if header is None:
        return None
    magic, = struct.unpack("<I", header[:4])
    if magic != MAINNET_MAGIC:
        return None
    cmd = header[4:16].rstrip(b"\x00").decode("ascii", errors="replace")
    length, = struct.unpack("<I", header[16:20])
    cksum = header[20:24]
    if length > max_payload:
        return None
    payload = _recvall(sock, length, timeout)
    if payload is None:
        return None
    if _checksum(payload) != cksum:
        return None
    return cmd, payload

def _ip_to_netaddr(ip: str) -> bytes:
    # Bitcoin "net_addr" uses 16-byte IP (IPv6), IPv4 mapped: ::ffff:a.b.c.d
    try:
        socket.inet_pton(socket.AF_INET6, ip)
        return socket.inet_pton(socket.AF_INET6, ip)
    except OSError:
        v4 = socket.inet_pton(socket.AF_INET, ip)
        return b"\x00" * 10 + b"\xff\xff" + v4

def _build_net_addr(services: int, ip: str, port: int) -> bytes:
    return struct.pack("<Q", services) + _ip_to_netaddr(ip) + struct.pack(">H", port)

def build_version_payload(
    addr_recv_ip: str,
    addr_recv_port: int,
    addr_from_ip: str,
    addr_from_port: int,
    user_agent: bytes = b"/AckProbing:1.0/",
    start_height: int = 0,
    relay: bool = False,
) -> bytes:
    """
    Bitcoin Core-style version payload.
    Fields (mainnet):
      int32 version
      uint64 services
      int64 timestamp
      net_addr addr_recv (services + ip + port)
      net_addr addr_from (services + ip + port)
      uint64 nonce
      var_str user_agent
      int32 start_height
      bool relay (since v70001)
    """
    version = 70016
    services = 0
    timestamp = int(time.time())
    nonce = random.getrandbits(64)

    payload = b""
    payload += struct.pack("<i", version)
    payload += struct.pack("<Q", services)
    payload += struct.pack("<q", timestamp)
    payload += _build_net_addr(services, addr_recv_ip, addr_recv_port)
    payload += _build_net_addr(services, addr_from_ip, addr_from_port)
    payload += struct.pack("<Q", nonce)
    payload += _pack_varstr(user_agent)
    payload += struct.pack("<i", start_height)
    payload += struct.pack("<?", relay)
    return payload

def build_verack() -> bytes:
    return _build_msg("verack", b"")

def build_ping(nonce: Optional[int] = None) -> bytes:
    if nonce is None:
        nonce = random.getrandbits(64)
    return _build_msg("ping", struct.pack("<Q", nonce))


# -----------------------------
# TCP measurement helpers
# -----------------------------

def _tcp_payload_len(pkt) -> int:
    try:
        return len(bytes(pkt[TCP].payload))
    except Exception:
        return 0

def _seq_add(a: int, b: int) -> int:
    return (a + b) % 0x100000000

def _next_seq_from_pkt(pkt) -> int:
    """
    For packets from peer -> us:
      their_next_seq = pkt.seq + payload_len (+1 if SYN/FIN)
    """
    their_seq = int(pkt[TCP].seq)
    plen = _tcp_payload_len(pkt)
    inc = plen
    flags = pkt[TCP].flags
    # SYN/FIN each consumes one sequence number
    if flags & 0x02:  # SYN
        inc += 1
    if flags & 0x01:  # FIN
        inc += 1
    return _seq_add(their_seq, inc)

def _is_ack(pkt) -> bool:
    return (pkt[TCP].flags & 0x10) != 0

def _is_rst(pkt) -> bool:
    return (pkt[TCP].flags & 0x04) != 0


@dataclass
class ProbeConfig:
    iface: str
    target_ip: str
    target_port: int
    rounds: int
    probes_per_round: int
    inter_probe_ms: int
    sniff_window_ms: int
    conn_timeout: float
    sniff_timeout: float


@dataclass
class RoundResult:
    round_idx: int
    ok: bool
    reason: str
    local_ip: Optional[str] = None
    local_port: Optional[int] = None
    our_next_seq: Optional[int] = None
    their_next_seq: Optional[int] = None
    sent_probes: int = 0
    challenge_acks: int = 0
    rsts: int = 0


class AckProber:
    def __init__(self, cfg: ProbeConfig):
        self.cfg = cfg
        self.sock: Optional[socket.socket] = None
        self.local_ip: Optional[str] = None
        self.local_port: Optional[int] = None

        # State synced via sniff
        self.our_next_seq: Optional[int] = None   # what peer expects next from us
        self.their_next_seq: Optional[int] = None # what we should ACK as next from peer

    def _connect_and_handshake(self) -> Tuple[bool, str]:
        """
        Establish TCP and complete Bitcoin version/verack handshake.
        """
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(self.cfg.conn_timeout)
            s.connect((self.cfg.target_ip, self.cfg.target_port))
            self.sock = s
            self.local_ip, self.local_port = s.getsockname()[0], s.getsockname()[1]
        except Exception as e:
            return False, f"TCP_CONNECT_FAILED: {e}"

        # Send version (with real addr info)
        try:
            vpayload = build_version_payload(
                addr_recv_ip=self.cfg.target_ip,
                addr_recv_port=self.cfg.target_port,
                addr_from_ip=self.local_ip,
                addr_from_port=self.local_port,
                user_agent=b"/AckProbing:1.0/",
                start_height=0,
                relay=False,
            )
            self.sock.sendall(_build_msg("version", vpayload))
        except Exception as e:
            return False, f"SEND_VERSION_FAILED: {e}"

        # Handshake loop
        got_version = False
        got_verack = False
        sent_verack = False

        deadline = time.time() + self.cfg.conn_timeout
        while time.time() < deadline and (not got_version or not got_verack):
            remaining = max(0.1, deadline - time.time())
            msg = _recv_msg(self.sock, timeout=remaining)
            if msg is None:
                break
            cmd, payload = msg
            if cmd == "version":
                got_version = True
                if not sent_verack:
                    try:
                        self.sock.sendall(build_verack())
                        sent_verack = True
                    except Exception as e:
                        return False, f"SEND_VERACK_FAILED: {e}"
            elif cmd == "verack":
                got_verack = True
            else:
                # ignore other messages
                pass

        # Some peers may send verack before version is parsed by us, but generally both appear quickly.
        if not sent_verack:
            try:
                self.sock.sendall(build_verack())
                sent_verack = True
            except Exception as e:
                return False, f"SEND_VERACK_FAILED_LATE: {e}"

        if not got_version:
            return False, "HANDSHAKE_FAILED_NO_VERSION"
        if not got_verack:
            return False, "HANDSHAKE_FAILED_NO_VERACK"
        return True, "OK"

    def _prime_traffic(self) -> bool:
        """
        Send a ping to encourage immediate TCP traffic for state sync sniffing.
        """
        try:
            self.sock.sendall(build_ping())
            return True
        except Exception:
            return False

    def _sniff_one_peer_packet(self) -> Optional[object]:
        """
        Sniff a packet from target -> us on the established 5-tuple.
        """
        if self.local_port is None:
            return None
        bpf = f"tcp and src host {self.cfg.target_ip} and dst port {self.local_port}"
        pkts = sniff(
            iface=self.cfg.iface,
            filter=bpf,
            timeout=self.cfg.sniff_timeout,
            count=1,
            store=True,
        )
        if not pkts:
            return None
        return pkts[0]

    def _sync_tcp_state(self) -> Tuple[bool, str]:
        """
        Sync (our_next_seq, their_next_seq) from one sniffed peer packet.
        """
        if not self._prime_traffic():
            # Even if ping fails, attempt sniff; maybe peer sends something anyway.
            pass

        pkt = self._sniff_one_peer_packet()
        if pkt is None or TCP not in pkt:
            return False, "SYNC_FAILED_NO_PACKET"

        if _is_rst(pkt):
            return False, "SYNC_FAILED_GOT_RST"

        # For a packet peer -> us:
        # - pkt.ack acknowledges bytes we sent. This is what peer expects next from us.
        # - their_next_seq = pkt.seq + payload_len (+ syn/fin adjustments).
        self.our_next_seq = int(pkt[TCP].ack) & 0xFFFFFFFF
        self.their_next_seq = _next_seq_from_pkt(pkt) & 0xFFFFFFFF

        return True, "OK"

    def _build_out_of_window_ack(self) -> int:
        """
        Choose an ACK number that is far outside the expected window.
        Use +2^31 offset mod 2^32 (common choice).
        """
        assert self.their_next_seq is not None
        return _seq_add(self.their_next_seq, 0x80000000)

    def _send_probe_once(self, probe_ack: int) -> bool:
        """
        Send a forged pure ACK segment on the same 5-tuple.
        """
        if self.local_ip is None or self.local_port is None:
            return False
        if self.our_next_seq is None:
            return False

        pkt = (
            IP(src=self.local_ip, dst=self.cfg.target_ip) /
            TCP(
                sport=self.local_port,
                dport=self.cfg.target_port,
                flags="A",
                seq=int(self.our_next_seq) & 0xFFFFFFFF,
                ack=int(probe_ack) & 0xFFFFFFFF,
                window=1024
            )
        )
        try:
            send(pkt, iface=self.cfg.iface, verbose=False)
            return True
        except Exception:
            return False

    def _sniff_responses(self, expected_ack: int, window_ms: int) -> Tuple[int, int]:
        """
        Sniff and count:
          - challenge ACKs: ACK-only packets from peer -> us that ACK our_next_seq (expected_ack),
            with zero payload (typical for challenge ACK).
          - RST packets (for diagnostics).
        """
        if self.local_port is None:
            return 0, 0

        bpf = f"tcp and src host {self.cfg.target_ip} and dst port {self.local_port}"
        timeout_s = max(0.001, window_ms / 1000.0)

        pkts = sniff(
            iface=self.cfg.iface,
            filter=bpf,
            timeout=timeout_s,
            store=True,
        )

        chal = 0
        rsts = 0
        for p in pkts:
            if TCP not in p:
                continue
            if _is_rst(p):
                rsts += 1
                continue
            if not _is_ack(p):
                continue
            # Strict-ish filter:
            # - ACK flag set
            # - ack == expected_ack (peer acknowledging our seq)
            # - payload length == 0 (control ack)
            if (int(p[TCP].ack) & 0xFFFFFFFF) == (expected_ack & 0xFFFFFFFF) and _tcp_payload_len(p) == 0:
                chal += 1

        return chal, rsts

    def close(self):
        try:
            if self.sock is not None:
                self.sock.close()
        except Exception:
            pass
        self.sock = None

    def run_round(self, round_idx: int) -> RoundResult:
        rr = RoundResult(round_idx=round_idx, ok=False, reason="INIT")

        ok, reason = self._connect_and_handshake()
        rr.local_ip, rr.local_port = self.local_ip, self.local_port
        if not ok:
            rr.reason = reason
            self.close()
            return rr

        ok, reason = self._sync_tcp_state()
        rr.our_next_seq, rr.their_next_seq = self.our_next_seq, self.their_next_seq
        if not ok:
            rr.reason = reason
            self.close()
            return rr

        # Print iptables hint once per round (useful when running as artifact)
        if rr.local_ip and rr.local_port:
            print(
                "[HINT] If results are unstable due to kernel TCP interference, consider dropping outbound RST for this 5-tuple:\n"
                f"  sudo iptables -A OUTPUT -p tcp -s {rr.local_ip} --sport {rr.local_port} "
                f"-d {self.cfg.target_ip} --dport {self.cfg.target_port} --tcp-flags RST RST -j DROP\n"
                "  (Remember to remove it after: sudo iptables -D OUTPUT <same rule>)\n"
            )

        probe_ack = self._build_out_of_window_ack()
        expected_ack = int(self.our_next_seq) & 0xFFFFFFFF

        sent = 0
        chal_total = 0
        rsts_total = 0

        for _ in range(self.cfg.probes_per_round):
            if self._send_probe_once(probe_ack):
                sent += 1
            # Sniff in a short window right after each probe
            chal, rsts = self._sniff_responses(expected_ack=expected_ack, window_ms=self.cfg.sniff_window_ms)
            chal_total += chal
            rsts_total += rsts
            time.sleep(max(0.0, self.cfg.inter_probe_ms / 1000.0))

        rr.sent_probes = sent
        rr.challenge_acks = chal_total
        rr.rsts = rsts_total
        rr.ok = True
        rr.reason = "OK"
        self.close()
        return rr


def parse_args():
    ap = argparse.ArgumentParser(description="TCP Challenge-ACK probing with Bitcoin-handshake stabilization.")
    ap.add_argument("--iface", required=True, help="Sniff/send interface name (e.g., eth0, wlan0).")
    ap.add_argument("--target-ip", required=True, help="Target peer IPv4 address.")
    ap.add_argument("--target-port", type=int, default=8333, help="Target peer TCP port (default: 8333).")
    ap.add_argument("--rounds", type=int, default=5, help="Number of rounds (default: 5).")
    ap.add_argument("--probes-per-round", type=int, default=3, help="Injected ACKs per round (default: 3).")
    ap.add_argument("--inter-probe-ms", type=int, default=50, help="Sleep between probes (default: 50ms).")
    ap.add_argument("--sniff-window-ms", type=int, default=120, help="Sniff window after each probe (default: 120ms).")
    ap.add_argument("--conn-timeout", type=float, default=3.0, help="TCP+handshake timeout seconds (default: 3.0).")
    ap.add_argument("--sniff-timeout", type=float, default=2.0, help="Sniff timeout for state sync (default: 2.0).")
    ap.add_argument("--out", default="", help="Optional output path (CSV).")
    return ap.parse_args()


def main():
    args = parse_args()

    # Basic sanity checks
    if os.geteuid() != 0:
        print("ERROR: This script requires root (raw sockets + sniff). Run with sudo.")
        sys.exit(1)
    if not (1 <= args.target_port <= 65535):
        print("ERROR: invalid target port.")
        sys.exit(1)

    cfg = ProbeConfig(
        iface=args.iface,
        target_ip=args.target_ip,
        target_port=args.target_port,
        rounds=args.rounds,
        probes_per_round=args.probes_per_round,
        inter_probe_ms=args.inter_probe_ms,
        sniff_window_ms=args.sniff_window_ms,
        conn_timeout=args.conn_timeout,
        sniff_timeout=args.sniff_timeout,
    )

    prober = AckProber(cfg)

    results: List[RoundResult] = []
    print(f"[INFO] Target = {cfg.target_ip}:{cfg.target_port}, iface={cfg.iface}")
    print(f"[INFO] rounds={cfg.rounds}, probes/round={cfg.probes_per_round}, sniff_window={cfg.sniff_window_ms}ms\n")

    for r in range(cfg.rounds):
        rr = prober.run_round(r)
        results.append(rr)
        if rr.ok:
            print(
                f"[ROUND {r}] OK  local={rr.local_ip}:{rr.local_port} "
                f"our_next_seq={rr.our_next_seq} their_next_seq={rr.their_next_seq} "
                f"sent={rr.sent_probes} challenge_acks={rr.challenge_acks} rsts={rr.rsts}"
            )
        else:
            print(f"[ROUND {r}] FAIL reason={rr.reason}")

    # Optional CSV output
    if args.out:
        import csv
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "round", "ok", "reason", "local_ip", "local_port",
                "our_next_seq", "their_next_seq", "sent_probes",
                "challenge_acks", "rsts"
            ])
            for rr in results:
                w.writerow([
                    rr.round_idx, int(rr.ok), rr.reason, rr.local_ip, rr.local_port,
                    rr.our_next_seq, rr.their_next_seq, rr.sent_probes,
                    rr.challenge_acks, rr.rsts
                ])
        print(f"\n[INFO] Wrote CSV to: {args.out}")

    # Summary
    ok_rounds = [x for x in results if x.ok]
    total_chal = sum(x.challenge_acks for x in ok_rounds)
    total_sent = sum(x.sent_probes for x in ok_rounds)
    print("\n[SUMMARY]")
    print(f"  ok_rounds={len(ok_rounds)}/{len(results)}")
    print(f"  total_sent_probes={total_sent}")
    print(f"  total_challenge_acks={total_chal}")
    if total_sent > 0:
        print(f"  challenge_acks_per_probe={total_chal / total_sent:.3f}")


if __name__ == "__main__":
    main()
