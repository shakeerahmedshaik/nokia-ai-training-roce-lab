#!/usr/bin/env bash
# Usage: bash run.sh <script.py> [script args...] [--wait]
# Starts rank 0 on worker1 (MASTER, TCP bootstrap store) and rank 1 on worker2.
# Logs go to /tmp/<script>.log in each worker.
set -euo pipefail

LAB=clab-ai-training
script=$1; shift
wait_done=0
args=()
for a in "$@"; do
  if [ "$a" = "--wait" ]; then wait_done=1; else args+=("$a"); fi
done
name=$(basename "$script" .py)

common=(-e MASTER_ADDR=10.10.1.2 -e MASTER_PORT=29500 -e WORLD_SIZE=2 -e GLOO_SOCKET_IFNAME=eth1)
rank=0
for w in worker1 worker2; do
  docker exec -d "${common[@]}" -e RANK=$rank -e ROCE_DEV=rxe_w$((rank + 1)) "$LAB-$w" \
    sh -c "exec /usr/bin/python3 -u /lab/app/$script ${args[*]:-} > /tmp/$name.log 2>&1"
  rank=$((rank + 1))
done

if [ "$wait_done" = 1 ]; then
  sleep 1
  for w in worker1 worker2; do
    while docker exec "$LAB-$w" pgrep -f "/lab/app/$script" >/dev/null; do sleep 1; done
  done
else
  sleep 3
fi

for w in worker1 worker2; do
  echo "===== $w /tmp/$name.log"
  docker exec "$LAB-$w" cat "/tmp/$name.log"
done
