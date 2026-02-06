

import argparse
import ipaddress
import re
import subprocess
from scapy.all import (
    ARP,
    Ether,
    conf,
    get_if_addr,
    srp,
)

MAC_RE = re.compile(r"^[0-9a-fA-F]{2}(:[0-9a-fA-F]{2}){5}$")


def norm_mac(mac: str) -> str:
    return mac.lower()


def get_ipv4_network(iface: str) -> ipaddress.IPv4Network:
    ip = get_if_addr(iface)

    if not ip or ip == "0.0.0.0":
        raise RuntimeError(f"Interface {iface} has no IPv4 address.")

    mask = None
    try:
        output = subprocess.check_output(f"ifconfig {iface}", shell=True).decode()
        match = re.search(r"netmask (0x[0-9a-fA-F]+)", output)
        if match:
            hex_mask = match.group(1)
            parts = [int(hex_mask[i:i+2], 16) for i in range(2, 10, 2)]
            mask = ".".join(map(str, parts))
    except Exception as e:
        print(f"[-] Failed to retrieve netmask via shell: {e}")

    if not mask:
        try:
            for addr in conf.ifaces.get(iface).ips[4]:
                if addr[0] == ip:
                    mask = addr[1]
        except:
            pass

    if not mask:
        print("[!] Failed to determine netmask, falling back to 255.255.255.0")
        mask = "255.255.255.0"

    print(f"[*] Determined IP: {ip}, Netmask: {mask}")
    return ipaddress.IPv4Network(f"{ip}/{mask}", strict=False)


def arp_scan_subnet(iface: str, net: ipaddress.IPv4Network, timeout: float, retry: int):
    pkt = Ether(dst="ff:ff:ff:ff:ff:ff") / ARP(pdst=str(net))
    answered, _ = srp(
        pkt,
        iface=iface,
        timeout=timeout,
        retry=retry,
        inter=0.0,
        verbose=False,
    )

    results = []
    for _, rcv in answered:
        if rcv.haslayer(ARP):
            results.append((rcv[ARP].psrc, norm_mac(rcv[ARP].hwsrc)))

    uniq = {}
    for ip, mac in results:
        uniq[ip] = mac

    return sorted(uniq.items(), key=lambda x: ipaddress.IPv4Address(x[0]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--iface", required=True)
    parser.add_argument("--victim-mac", default=None)
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument("--retry", type=int, default=1)
    parser.add_argument("--show-all", action="store_true")
    args = parser.parse_args()

    if args.victim_mac:
        if not MAC_RE.match(args.victim_mac):
            raise SystemExit("Invalid MAC format (expected aa:bb:cc:dd:ee:ff)")
        victim_mac = norm_mac(args.victim_mac)
    else:
        victim_mac = None

    conf.verb = 0

    net = get_ipv4_network(args.iface)
    print(f"[+] Interface: {args.iface}")
    print(f"[+] Local subnet: {net.network_address}/{net.prefixlen} (size={net.num_addresses})")
    print(f"[+] Broadcasting ARP who-has over {net} ...")

    hosts = arp_scan_subnet(args.iface, net, args.timeout, args.retry)
    print(f"[+] Received {len(hosts)} ARP replies.")

    if args.show_all:
        for ip, mac in hosts:
            print(f"{ip:15s} {mac}")

    if victim_mac:
        matches = [(ip, mac) for ip, mac in hosts if mac == victim_mac]
        if matches:
            for ip, _ in matches:
                print(f"[✓] victim_mac={victim_mac} -> victim_ip={ip}")
        else:
            print("[!] Victim MAC not observed in ARP replies.")
            print("Possible causes:")
            print("- Client isolation enabled")
            print("- Victim not responding to ARP")
            print("- Different subnet or VLAN")
            print("- MAC randomization")
            print("Mitigations:")
            print("- Ensure same SSID/VLAN and isolation disabled")
            print("- Increase timeout or retry count")
    else:
        print("[i] No victim MAC specified.")

    print("Done.")


if __name__ == "__main__":
    main()
