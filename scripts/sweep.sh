#!/usr/bin/env bash
# Section 8 sweeps, one variable at a time, RoCEv2 transport. Each run captures headers on
# worker1 eth1 (original frame lengths are kept) and prints one summary row.
set -euo pipefail

LAB=clab-ai-training
HERE=$(cd "$(dirname "$0")/.." && pwd)
cd "$HERE"

discards() {
  for n in leaf1 leaf2; do
    docker exec "$LAB-$n" sr_cli -d "info from state interface ethernet-1/10 statistics in-discarded-packets" | awk '{print $2}'
  done | awk '{s+=$1} END {print s}'
}

row() {
  local label=$1; shift
  local d0 d1
  d0=$(discards)
  bash scripts/capture.sh start sweep --snap 96 >/dev/null
  bash run.sh train.py "$@" --wait > /tmp/sweep-run.txt
  bash scripts/capture.sh stop sweep >/dev/null
  d1=$(discards)
  local log params gbytes step_ms ar_ms frames bytes ck
  log=$(sed -n '/worker1/,/worker2/p' /tmp/sweep-run.txt)
  params=$(grep -oP 'params=\K[0-9]+' <<<"$log" | head -1)
  gbytes=$(grep -oP 'gradient_bytes_FP32=\K[0-9]+' <<<"$log" | head -1)
  step_ms=$(grep -oP 'step=[0-9]+ .*compute_ms=\K[0-9.]+' <<<"$log" | awk '{s+=$1;n++} END {printf "%.1f", n ? s/n : 0}')
  ar_ms=$(grep -oP 'step=[0-9]+ .*allreduce_ms=\K[0-9.]+' <<<"$log" | awk '{s+=$1;n++} END {printf "%.0f", n ? s/n : 0}')
  docker cp captures/sweep.pcap "$LAB-worker1:/tmp/sweep-a.pcap" >/dev/null
  read -r frames bytes < <(docker exec "$LAB-worker1" tshark -r /tmp/sweep-a.pcap -Y "udp.dstport==4791" \
    -T fields -e frame.len 2>/dev/null | awk '{n++; s+=$1} END {print n+0, s+0}')
  ck=$(grep -oP 'final model_checksum=\K\S+' /tmp/sweep-run.txt | sort -u | wc -l)
  printf '%-16s params=%-8s grad_MB=%-6.1f wire_MB=%-7.1f roce_frames=%-7s mean_compute_ms=%-6s mean_allreduce_ms=%-6s leaf_discards=%-6s checksums_match=%s\n' \
    "$label" "$params" "$(echo "$gbytes/1000000" | bc -l)" "$(echo "$bytes/1000000" | bc -l)" "$frames" \
    "$step_ms" "$ar_ms" "$((d1 - d0))" "$([ "$ck" = 1 ] && echo yes || echo NO)"
}

bash scripts/pace.sh off >/dev/null

echo "## hidden (features=1024, steps=20, batch=32)"
for h in 256 2048 4096; do row "hidden=$h" --steps 20 --features 1024 --hidden "$h"; done

echo "## steps (hidden=2048)"
for s in 1 20; do row "steps=$s" --steps "$s" --features 1024 --hidden 2048; done

echo "## batch (hidden=2048, steps=20)"
for b in 8 64; do row "batch=$b" --steps 20 --features 1024 --hidden 2048 --batch "$b"; done

echo "## sender pacing on both workers (hidden=2048, steps=20)"
for r in off 100mbit 60mbit; do
  bash scripts/pace.sh "$r" >/dev/null
  row "pace=$r" --steps 20 --features 1024 --hidden 2048
done
bash scripts/pace.sh off >/dev/null

echo "## netem delay on worker2 eth1 (hidden=2048, steps=20)"
for d in 0 5 20; do
  if [ "$d" = 0 ]; then bash scripts/netem.sh del >/dev/null; else bash scripts/netem.sh add "$d" >/dev/null; fi
  row "delay=${d}ms" --steps 20 --features 1024 --hidden 2048
done
bash scripts/netem.sh del >/dev/null
