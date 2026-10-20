"""NCCL-style ring AllReduce over RoCEv2 (Soft-RoCE / rxe) using libibverbs RC queue pairs.

Bootstrap (QP numbers, GIDs, rkeys) goes over a small TCP store, as NCCL bootstraps over sockets.
All gradient bytes move as RDMA WRITE-with-immediate on UDP/4791. Each channel is its own RC QP,
so its UDP source port differs and the fabric can ECMP-hash channels onto different spines.
Ring algorithm for world=2: reduce-scatter (send the peer's half, add the received half),
then all-gather (send the reduced own half). Each half is striped across the channels.
"""
import glob
import json
import os
import random
import time

import numpy as np
import pyverbs.enums as e
from pyverbs.addr import GID, AHAttr, GlobalRoute
from pyverbs.cq import CQ
from pyverbs.device import Context
from pyverbs.mr import MR
from pyverbs.pd import PD
from pyverbs.qp import QP, QPAttr, QPCap, QPInitAttr
from pyverbs.wr import SGE, RecvWR, SendWR

ROCE_DSCP = 26
ECT0 = 0b10
PORT = 1
RS, AG = 1, 2


def roce_v2_ipv4_gid_index(dev):
    base = f"/sys/class/infiniband/{dev}/ports/{PORT}"
    for path in sorted(glob.glob(f"{base}/gids/*"), key=lambda p: int(p.rsplit("/", 1)[1])):
        idx = int(path.rsplit("/", 1)[1])
        try:
            gid = open(path).read().strip()
            gtype = open(f"{base}/gid_attrs/types/{idx}").read().strip()
        except OSError:
            continue
        if gid.startswith("0000:0000:0000:0000:0000:ffff") and "v2" in gtype:
            return idx
    raise RuntimeError(f"no RoCE v2 IPv4 GID on {dev}")


class RoceRing:
    def __init__(self, store, rank, world, numel, dev, channels=2, mtu=e.IBV_MTU_4096,
                 tclass=(ROCE_DSCP << 2) | ECT0, log=print):
        assert world == 2, "this lab's ring is written for two ranks"
        self.rank, self.world, self.channels, self.log = rank, world, channels, log
        unit = 2 * channels
        self.numel = -(-numel // unit) * unit
        self.nbytes = self.numel * 4
        self.stripe = self.nbytes // unit

        self.ctx = Context(name=dev)
        self.pd = PD(self.ctx)
        self.cq = CQ(self.ctx, 4096)
        access = e.IBV_ACCESS_LOCAL_WRITE | e.IBV_ACCESS_REMOTE_WRITE
        self.buf = MR(self.pd, self.nbytes, access)
        self.rcv = MR(self.pd, self.nbytes, access)
        self.imm_sink = MR(self.pd, 64, e.IBV_ACCESS_LOCAL_WRITE)
        self.gid_index = roce_v2_ipv4_gid_index(dev)
        gid = self.ctx.query_gid(PORT, self.gid_index)

        cap = QPCap(max_send_wr=256, max_recv_wr=512, max_send_sge=1, max_recv_sge=1)
        self.qps = [QP(self.pd, QPInitAttr(qp_type=e.IBV_QPT_RC, scq=self.cq, rcq=self.cq, cap=cap))
                    for _ in range(channels)]
        psn = [random.randrange(1 << 24) for _ in range(channels)]
        me = dict(gid=str(gid), qpn=[q.qp_num for q in self.qps], psn=psn,
                  buf=[self.buf.buf, self.buf.rkey], rcv=[self.rcv.buf, self.rcv.rkey])
        store.set(f"roce/{rank}", json.dumps(me))
        peer = json.loads(store.get(f"roce/{1 - rank}"))
        self.peer = peer

        for i, qp in enumerate(self.qps):
            attr = QPAttr(port_num=PORT)
            attr.qp_access_flags = access
            qp.to_init(attr)
            gr = GlobalRoute(dgid=GID(peer["gid"]), sgid_index=self.gid_index,
                             hop_limit=64, traffic_class=tclass)
            attr.ah_attr = AHAttr(port_num=PORT, is_global=1, gr=gr)
            attr.path_mtu = mtu
            attr.dest_qp_num = peer["qpn"][i]
            attr.rq_psn = peer["psn"][i]
            attr.max_dest_rd_atomic = 1
            attr.min_rnr_timer = 12
            qp.to_rtr(attr)
            attr.sq_psn = psn[i]
            attr.timeout = 14
            attr.retry_cnt = 7
            attr.rnr_retry = 7
            attr.max_rd_atomic = 1
            qp.to_rts(attr)
            for _ in range(256):
                qp.post_recv(self._recv_wr())
        self.qpn_to_ch = {q.qp_num: i for i, q in enumerate(self.qps)}
        self.seq = 0
        store.set(f"roce/ready/{rank}", "1")
        store.get(f"roce/ready/{1 - rank}")
        log(f"roce dev={dev} gid_index={self.gid_index} gid={gid} qpn={me['qpn']} "
            f"peer_gid={peer['gid']} peer_qpn={peer['qpn']} channels={channels} "
            f"tclass=0x{tclass:02x} (DSCP {tclass >> 2}, ECN {tclass & 3})")

    def _recv_wr(self):
        return RecvWR(num_sge=1, sg=[SGE(self.imm_sink.buf, 64, self.imm_sink.lkey)])

    def _write(self, ch, src_off, peer_mr, dst_off, tag):
        qp = self.qps[ch]
        wr = SendWR(wr_id=tag, opcode=e.IBV_WR_RDMA_WRITE_WITH_IMM, num_sge=1,
                    sg=[SGE(self.buf.buf + src_off, self.stripe, self.buf.lkey)],
                    send_flags=e.IBV_SEND_SIGNALED)
        wr.imm_data = tag
        wr.set_wr_rdma(self.peer[peer_mr][1], self.peer[peer_mr][0] + dst_off)
        qp.post_send(wr)

    def _wait(self, sends, recvs, timeout=60.0):
        deadline = time.monotonic() + timeout
        while sends or recvs:
            n, wcs = self.cq.poll(num_entries=16)
            for wc in wcs[:n]:
                if wc.status != e.IBV_WC_SUCCESS:
                    raise RuntimeError(f"WC error status={wc.status} opcode={wc.opcode} qpn={wc.qp_num}")
                if wc.opcode == e.IBV_WC_RDMA_WRITE:
                    sends.discard(wc.wr_id)
                elif wc.opcode == e.IBV_WC_RECV_RDMA_WITH_IMM:
                    recvs.discard(wc.imm_data)
                    self.qps[self.qpn_to_ch[wc.qp_num]].post_recv(self._recv_wr())
            if not n and time.monotonic() > deadline:
                raise TimeoutError(f"pending sends={sends} recvs={recvs}")

    def _offset(self, half, ch):
        return (half * self.channels + ch) * self.stripe

    def _tag(self, phase, ch):
        return ((self.seq & 0xFFFFF) << 8) | (phase << 4) | ch

    def allreduce_(self, flat):
        """In-place average of a contiguous float32 tensor or ndarray across both ranks."""
        arr = flat.numpy() if hasattr(flat, "numpy") else flat
        n = arr.size
        data = np.zeros(self.numel, dtype=np.float32)
        data[:n] = arr.reshape(-1)
        self.buf.write(data.tobytes(), self.nbytes, 0)
        own, other = self.rank, 1 - self.rank
        stats = {}

        t0 = time.perf_counter()
        sends, recvs = set(), set()
        for ch in range(self.channels):
            off = self._offset(other, ch)
            tag = self._tag(RS, ch)
            self._write(ch, off, "rcv", off, tag)
            sends.add(tag)
            recvs.add(tag)
        self._wait(sends, recvs)
        lo, hi = self._offset(own, 0), self._offset(own + 1, 0)
        mine = np.frombuffer(self.buf.read(hi - lo, lo), dtype=np.float32)
        theirs = np.frombuffer(self.rcv.read(hi - lo, lo), dtype=np.float32)
        reduced = mine + theirs
        self.buf.write(reduced.tobytes(), hi - lo, lo)
        stats["reduce_scatter_ms"] = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        sends, recvs = set(), set()
        for ch in range(self.channels):
            off = self._offset(own, ch)
            tag = self._tag(AG, ch)
            self._write(ch, off, "buf", off, tag)
            sends.add(tag)
            recvs.add(tag)
        self._wait(sends, recvs)
        stats["all_gather_ms"] = (time.perf_counter() - t1) * 1000
        self.seq += 1

        out = np.frombuffer(self.buf.read(self.nbytes, 0), dtype=np.float32)[:n] / self.world
        arr.reshape(-1)[:] = out
        stats["wire_payload_bytes_sent"] = 2 * self.channels * self.stripe
        return stats
