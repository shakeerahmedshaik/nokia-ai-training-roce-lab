#!/usr/bin/env bash
# Usage: scripts/pace.sh <rate e.g. 100mbit> | off
# Token-bucket pacing on both workers' eth1, standing in for NIC rate control (DCQCN reaction point).
set -euo pipefail

LAB=clab-ai-training
for w in worker1 worker2; do
  if [ "$1" = off ]; then
    docker exec "$LAB-$w" tc qdisc del dev eth1 root 2>/dev/null || true
  else
    docker exec "$LAB-$w" tc qdisc replace dev eth1 root tbf rate "$1" burst 64kb latency 100ms
  fi
  printf '%s: ' "$w"; docker exec "$LAB-$w" tc qdisc show dev eth1 | head -1
done
