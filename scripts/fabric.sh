#!/usr/bin/env bash
# Usage: scripts/fabric.sh [counters]
# Prints BGP sessions, ECMP routes for both worker subnets, and spine interface counters.
set -euo pipefail

LAB=clab-ai-training
srl() { docker exec "$LAB-$1" sr_cli -d "$2"; }

counters() {
  for s in spine1 spine2; do
    for p in ethernet-1/1 ethernet-1/2; do
      printf '%-7s %-13s ' "$s" "$p"
      srl "$s" "info from state interface $p statistics" |
        awk '/in-octets|out-octets|in-packets|out-packets/ && !/discarded|error/ {printf "%s=%s ", $1, $2} END {print ""}'
    done
  done
}

if [ "${1:-}" = counters ]; then counters; exit 0; fi

for n in leaf1 leaf2; do
  echo "### $n BGP"
  srl "$n" "show network-instance default protocols bgp neighbor" | grep -E 'established|configured sessions' || true
done
echo "### leaf1 route to worker2 subnet"
srl leaf1 "show network-instance default ipv4 route" | grep -A1 '^10.10.2.0/24'
echo "### leaf2 route to worker1 subnet"
srl leaf2 "show network-instance default ipv4 route" | grep -A1 '^10.10.1.0/24'
echo "### spine counters"
counters
