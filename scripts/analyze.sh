#!/usr/bin/env bash
# Usage: scripts/analyze.sh captures/<name>.pcap
# Runs tshark inside worker1 and prints the packet worksheet: conversations, frame sizes,
# I/O over time, RoCEv2 BTH opcode/QP/DSCP/ECN breakdown, TTL and totals.
set -eu

LAB=clab-ai-training
pcap=$1
f=/tmp/analyze-$(basename "$pcap")
docker cp "$pcap" "$LAB-worker1:$f" >/dev/null
ts() { docker exec "$LAB-worker1" tshark -r "$f" "$@" 2>/dev/null; }

echo "### Totals"
ts -q -z io,stat,0,"frame","udp.dstport==4791","tcp" | sed -n '/Interval/,$p' | grep -v '^=*$'

echo; echo "### UDP conversations (RoCEv2 = port 4791, one source port per QP)"
ts -q -z conv,udp | sed -n '/<->/p' | head -20
echo; echo "### TCP conversations (bootstrap store only)"
ts -q -z conv,tcp | sed -n '/<->/p' | head -20

echo; echo "### Frame length distribution"
ts -q -z plen,tree | sed -n '/Packet Lengths/,$p' | grep -vE '^=+$' | head -20

echo; echo "### RoCEv2 BTH opcode by destination QP (count, opcode, dest QP)"
ts -Y "udp.dstport==4791" -T fields -e infiniband.bth.opcode -e infiniband.bth.destqp |
  sort | uniq -c | sort -rn | head -20

echo; echo "### IP DSCP / ECN / TTL on RoCEv2 frames (count, dscp, ecn, ttl, src)"
ts -Y "udp.dstport==4791" -T fields -e ip.dsfield.dscp -e ip.dsfield.ecn -e ip.ttl -e ip.src |
  sort | uniq -c | sort -rn | head -10

echo; echo "### First RDMA WRITE packets (time, src, udp.srcport, opcode, destqp, psn, reth.dmalen)"
ts -Y "infiniband.bth.opcode in {6, 7, 8, 10, 11}" -T fields -e frame.time_relative -e ip.src \
  -e udp.srcport -e infiniband.bth.opcode -e infiniband.bth.destqp -e infiniband.bth.psn \
  -e infiniband.reth.dmalen | head -8

echo; echo "### I/O graph, ${IO_INTERVAL:-0.5} s buckets (frames and bytes, RoCEv2 only)"
ts -q -z io,stat,"${IO_INTERVAL:-0.5}","udp.dstport==4791" | sed -n '/Interval/,$p' | grep -vE '^=+$' | head -70
docker exec "$LAB-worker1" rm -f "$f"
