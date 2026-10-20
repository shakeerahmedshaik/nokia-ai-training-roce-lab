#!/usr/bin/env bash
# Spine-link failure during training: leaf1 ethernet-1/1 (to spine1) goes down at T+DOWN_AT s and
# comes back at T+UP_AT s. The RC QP hashed onto spine1 loses in-flight frames and recovers via
# go-back-N once BGP withdraws the path and ECMP rehashes the flow onto spine2.
set -euo pipefail

LAB=clab-ai-training
HERE=$(cd "$(dirname "$0")/.." && pwd)
cd "$HERE"
STEPS=${STEPS:-40}
DOWN_AT=${DOWN_AT:-15}
UP_AT=${UP_AT:-30}

link() {
  printf 'set / interface ethernet-1/1 admin-state %s\ncommit now\n' "$1" |
    docker exec -i "$LAB-leaf1" sr_cli -e >/dev/null
  echo "$(date +%T.%3N) leaf1 ethernet-1/1 admin-state $1"
}

bash run.sh train.py --steps "$STEPS" --features 1024 --hidden 2048 --wait > /tmp/failover-run.txt &
run=$!
t0=$(date +%s)
sleep "$DOWN_AT"; link disable
sleep 3
docker exec "$LAB-leaf1" sr_cli -d "show network-instance default ipv4 route" | grep -A1 '^10.10.2.0/24'
sleep $((UP_AT - DOWN_AT - 3)); link enable
wait $run
echo "run finished after $(( $(date +%s) - t0 )) s"
grep -E 'rank=0 step=' /tmp/failover-run.txt | grep -oP 'step=\d+|allreduce_ms=[0-9.]+' | paste - - | column -t
grep -oP 'final model_checksum=\S+' /tmp/failover-run.txt | sort | uniq -c
sleep 5
docker exec "$LAB-leaf1" sr_cli -d "show network-instance default ipv4 route" | grep -A1 '^10.10.2.0/24'
