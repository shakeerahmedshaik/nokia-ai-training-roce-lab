#!/usr/bin/env bash
# Usage: scripts/capture.sh start <name> [--snap N]
#        scripts/capture.sh stop  <name>
# Captures worker1 eth1 (full frames by default) and copies the pcap to ./captures on stop.
# SR Linux transit ports are not captured: its datapath forwards most frames without passing
# them to the kernel packet tap, so spine-side placement is measured with scripts/fabric.sh counters.
set -euo pipefail

LAB=clab-ai-training
HERE=$(cd "$(dirname "$0")/.." && pwd)
action=$1 name=$2; shift 2
snap=0
while [ $# -gt 0 ]; do
  case $1 in
    --snap) snap=$2; shift ;;
  esac
  shift
done

case $action in
  start)
    docker exec "$LAB-worker1" rm -f "/tmp/$name.pcap"
    docker exec -d "$LAB-worker1" tcpdump -i eth1 -nn -s "$snap" -U -w "/tmp/$name.pcap"
    sleep 1
    echo "capture $name started"
    ;;
  stop)
    sleep 1
    docker exec "$LAB-worker1" pkill -INT tcpdump || true
    sleep 1
    mkdir -p "$HERE/captures"
    docker cp "$LAB-worker1:/tmp/$name.pcap" "$HERE/captures/$name.pcap" >/dev/null
    ls -la "$HERE/captures/$name.pcap"
    ;;
esac
