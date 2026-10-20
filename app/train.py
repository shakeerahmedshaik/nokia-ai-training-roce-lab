"""Experiment B: data-parallel MLP training. Gradients are averaged every step with an
NCCL-style ring AllReduce over RoCEv2 (default) or with PyTorch DDP over Gloo/TCP (--transport gloo)."""
import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn

p = argparse.ArgumentParser()
p.add_argument("--steps", type=int, default=20)
p.add_argument("--features", type=int, default=1024)
p.add_argument("--hidden", type=int, default=2048)
p.add_argument("--batch", type=int, default=32)
p.add_argument("--lr", type=float, default=0.003)
p.add_argument("--channels", type=int, default=2)
p.add_argument("--transport", choices=["roce", "gloo"], default="roce")
args = p.parse_args()

rank = int(os.environ["RANK"])
world = int(os.environ["WORLD_SIZE"])
torch.set_num_threads(2)


def log(msg):
    print(f"rank={rank} {msg}", flush=True)


def grad_checksum(m):
    return sum(q.grad.double().sum().item() for q in m.parameters())


def model_checksum(m):
    return sum(q.detach().double().sum().item() for q in m.parameters())


torch.manual_seed(0)
model = nn.Sequential(nn.Linear(args.features, args.hidden), nn.ReLU(), nn.Linear(args.hidden, 1))
w_true = torch.randn(args.features, 1)
params = sum(q.numel() for q in model.parameters())
log(f"transport={args.transport} params={params} gradient_bytes_FP32={params * 4} "
    f"batch={args.batch} steps={args.steps}")

if args.transport == "roce":
    from roce_ring import RoceRing

    store = dist.TCPStore(os.environ["MASTER_ADDR"], int(os.environ["MASTER_PORT"]), world,
                          is_master=rank == 0)
    ring = RoceRing(store, rank, world, params, os.environ["ROCE_DEV"], channels=args.channels, log=log)
    flat = torch.zeros(params)

    def sync_grads(net):
        torch.cat([q.grad.flatten() for q in net.parameters()], out=flat)
        stats = ring.allreduce_(flat)
        off = 0
        for q in net.parameters():
            q.grad.copy_(flat[off:off + q.numel()].view_as(q.grad))
            off += q.numel()
        return stats

    net = model
else:
    from torch.nn.parallel import DistributedDataParallel as DDP

    dist.init_process_group("gloo", init_method="env://")
    net = DDP(model)

    def sync_grads(_net):
        return {}

opt = torch.optim.SGD(net.parameters(), lr=args.lr)
loss_fn = nn.MSELoss()
data = torch.Generator().manual_seed(1000 + rank)


def batch():
    x = torch.randn(args.batch, args.features, generator=data)
    return x, x @ w_true / args.features**0.5


eval_gen = torch.Generator().manual_seed(7)
x_eval = torch.randn(256, args.features, generator=eval_gen)
y_eval = x_eval @ w_true / args.features**0.5


def eval_loss():
    with torch.no_grad():
        return loss_fn(model(x_eval), y_eval).item()


log(f"initial model_checksum={model_checksum(model):.10f} eval_loss={eval_loss():.6f}")

x0, y0 = batch()
if args.transport == "gloo":
    with net.no_sync():
        loss_fn(net(x0), y0).backward()
else:
    loss_fn(net(x0), y0).backward()
log(f"probe local_grad_checksum={grad_checksum(model):.10f}")
opt.zero_grad(set_to_none=False)

for step in range(args.steps):
    x, y = (x0, y0) if step == 0 else batch()
    t0 = time.perf_counter()
    loss = loss_fn(net(x), y)
    loss.backward()
    t1 = time.perf_counter()
    stats = sync_grads(net)
    t2 = time.perf_counter()
    synced = grad_checksum(model)
    opt.step()
    opt.zero_grad(set_to_none=False)
    t3 = time.perf_counter()
    extra = " ".join(f"{k}={v:.1f}" for k, v in stats.items() if k.endswith("_ms"))
    log(f"step={step} loss={loss.item():.6f} synced_grad_checksum={synced:.10f} "
        f"compute_ms={(t1 - t0) * 1000:.1f} allreduce_ms={(t2 - t1) * 1000:.1f} {extra} "
        f"elapsed_ms={(t3 - t0) * 1000:.1f}")

log(f"final model_checksum={model_checksum(model):.10f} eval_loss={eval_loss():.6f}")
if args.transport == "roce":
    store.set(f"done/{rank}", "1")
    store.get(f"done/{1 - rank}")
else:
    dist.barrier()
    dist.destroy_process_group()
