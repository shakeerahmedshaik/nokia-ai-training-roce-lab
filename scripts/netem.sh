#!/usr/bin/env bash
# Usage: scripts/netem.sh add <delay_ms> | del | show   (applies to worker2 eth1 egress)
set -euo pipefail

C=clab-ai-training-worker2
case $1 in
  add) docker exec "$C" tc qdisc replace dev eth1 root netem delay "${2}ms" ;;
  del) docker exec "$C" tc qdisc del dev eth1 root 2>/dev/null || true ;;
esac
docker exec "$C" tc qdisc show dev eth1
