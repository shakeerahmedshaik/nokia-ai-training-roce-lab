"""Experiment A: scalar AllReduce. Default transport is RoCEv2 (rxe); --transport gloo uses TCP."""
import argparse
import os

import torch

p = argparse.ArgumentParser()
p.add_argument("--transport", choices=["roce", "gloo"], default="roce")
args = p.parse_args()

rank = int(os.environ["RANK"])
world = int(os.environ["WORLD_SIZE"])
t = torch.tensor([2.0 * (rank + 1)])
print(f"rank={rank} world={world} transport={args.transport} before={t.item()}", flush=True)

if args.transport == "roce":
    import torch.distributed as dist
    from roce_ring import RoceRing

    store = dist.TCPStore(os.environ["MASTER_ADDR"], int(os.environ["MASTER_PORT"]), world,
                          is_master=rank == 0)
    ring = RoceRing(store, rank, world, t.numel(), os.environ["ROCE_DEV"],
                    log=lambda m: print(f"rank={rank} {m}", flush=True))
    ring.allreduce_(t)
    print(f"rank={rank} after={t.item()}", flush=True)
    store.set(f"done/{rank}", "1")
    store.get(f"done/{1 - rank}")
else:
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method="env://")
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    print(f"rank={rank} sum={t.item()}", flush=True)
    t /= world
    print(f"rank={rank} after={t.item()}", flush=True)
    dist.barrier()
    dist.destroy_process_group()
