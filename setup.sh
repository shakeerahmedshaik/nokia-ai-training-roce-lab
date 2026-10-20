#!/usr/bin/env bash
# Prepare the workers (CPU PyTorch + Soft-RoCE) and verify the SR Linux Clos fabric.
# Host prerequisite: sudo modprobe rdma_rxe ib_uverbs. Intended for a fresh deployment.
set -euo pipefail

LAB=clab-ai-training
W1=$LAB-worker1
W2=$LAB-worker2
HERE=$(cd "$(dirname "$0")" && pwd)

srl() { docker exec "$LAB-$1" sr_cli -d "$2"; }

install_worker() {
  local c=$1
  echo "== $c: installing packages (apt rdma-core/pyverbs + CPU PyTorch)"
  docker exec "$c" sh -c 'export DEBIAN_FRONTEND=noninteractive;
    apt-get update -qq &&
    apt-get install -y -qq --no-install-recommends iproute2 iputils-ping tcpdump tshark ethtool procps \
      rdma-core ibverbs-providers ibverbs-utils perftest rdmacm-utils python3 python3-pip python3-pyverbs >/dev/null'
  # pyverbs comes from Debian for the system python3, so torch is installed for the same interpreter.
  docker exec "$c" /usr/bin/python3 -m pip install -q --no-cache-dir --break-system-packages \
    --root-user-action=ignore torch numpy --index-url https://download.pytorch.org/whl/cpu
}

config_worker() {
  local c=$1 addr=$2 gw=$3 dev=$4
  echo "== $c: eth1 $addr via $gw, Soft-RoCE device $dev"
  docker exec "$c" ip link set eth1 mtu 9000
  docker exec "$c" ip addr add "$addr" dev eth1
  docker exec "$c" ip route add 10.10.0.0/16 via "$gw" dev eth1
  # Wire-sized frames in captures, and valid L4 checksums once frames leave the veth for SR Linux.
  docker exec "$c" ethtool -K eth1 tso off gso off gro off tx off >/dev/null
  docker exec "$c" rdma link add "$dev" type rxe netdev eth1
}

# uverbs character devices appear on the host after the container started, so create them inside.
mk_uverbs() {
  local c=$1
  docker exec "$c" sh -c 'mkdir -p /dev/infiniband
    for d in /sys/class/infiniband_verbs/uverbs*; do
      n=/dev/infiniband/$(basename $d); [ -e $n ] && continue
      mknod $n c $(cut -d: -f1 $d/dev) $(cut -d: -f2 $d/dev)
    done'
}

install_worker "$W1" &
install_worker "$W2" &
wait

config_worker "$W1" 10.10.1.2/24 10.10.1.1 rxe_w1
config_worker "$W2" 10.10.2.2/24 10.10.2.1 rxe_w2
mk_uverbs "$W1"
mk_uverbs "$W2"
docker exec "$W1" rdma link show

echo "== waiting for 4 established BGP sessions"
for i in $(seq 1 60); do
  up=0
  for n in leaf1 leaf2; do
    c=$(srl "$n" "show network-instance default protocols bgp neighbor" 2>/dev/null | grep -E '\| +established +\|' | wc -l)
    up=$((up + c))
  done
  [ "$up" -ge 4 ] && break
  sleep 5
done
[ "$up" -ge 4 ] || { echo "FAIL: only $up/4 BGP sessions established"; exit 1; }
echo "   $up/4 sessions established"

echo "== RoCE QoS (DSCP 26 RoCE data class with ECN, DSCP 48 CNP strict priority)"
qos_port_cli() {
  local p=$1
  cat <<EOF
set / qos interfaces interface $p interface-ref interface $p
set / qos interfaces interface $p output scheduler scheduler-policy ROCE
set / qos interfaces interface $p output buffer-allocation-profile ROCE
set / qos interfaces interface $p output queues queue unicast-3 queue-management-profile ROCE-ECN
set / qos interfaces interface $p.0 interface-ref interface $p
set / qos interfaces interface $p.0 interface-ref subinterface 0
set / qos interfaces interface $p.0 input classifiers dscp-policy ROCE
EOF
}
for n in leaf1 leaf2 spine1 spine2; do
  ports="ethernet-1/1 ethernet-1/2"
  case $n in leaf*) ports="$ports ethernet-1/10" ;; esac
  { cat "$HERE/configs/qos-roce.cli"; for p in $ports; do qos_port_cli "$p"; done; echo "commit now"; } |
    docker exec -i "$LAB-$n" sr_cli -e >/tmp/qos-$n.out 2>&1 &&
    grep -qi "All changes have been committed" /tmp/qos-$n.out && echo "   $n: committed" ||
    { echo "   $n: QoS commit FAILED"; cat /tmp/qos-$n.out; }
done

echo "== ECMP check on leaf1 for 10.10.2.0/24"
rt=$(srl leaf1 "show network-instance default ipv4 route")
grep -A1 '^10.10.2.0/24' <<<"$rt"
nh=$(grep -A1 '^10.10.2.0/24' <<<"$rt" | grep -cE '192\.168\.1[12]\.1' || true)
[ "$nh" -ge 2 ] && echo "   2 ECMP next-hops installed" || echo "WARN: expected 2 next-hops, found $nh"

echo "== ping worker1 -> worker2"
docker exec "$W1" ping -c 3 -W 2 10.10.2.2
