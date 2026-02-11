# Multi-Node DP Replicas: Design Document

## Problem Statement

Serving large MoE models like **Kimi K2** (moonshotai/Kimi-K2-Instruct) requires
TP=8 and PP=2, totaling **16 GPUs per replica**. On 8-GPU nodes, a single replica
spans **2 nodes**. To maximize throughput on an 8-node (64 GPU) cluster, we want
**4 DP replicas**, each spanning 2 nodes.

```
┌─────────────────────────────────────────────────────────────────────┐
│  8 nodes × 8 GPUs = 64 GPUs                                        │
│                                                                     │
│  DP replica 0:  Node 0 (leader, head) + Node 1 (follower)          │
│  DP replica 1:  Node 2 (leader)       + Node 3 (follower)          │
│  DP replica 2:  Node 4 (leader)       + Node 5 (follower)          │
│  DP replica 3:  Node 6 (leader)       + Node 7 (follower)          │
│                                                                     │
│  Node 0 also hosts: API server, coordinator, internal load balancer │
└─────────────────────────────────────────────────────────────────────┘
```

**Before this change:** vLLM clamped `data_parallel_size_local = max(..., 1)`,
forcing every headless node to create an EngineCore. Follower nodes (which only
need to run GPU workers for the leader's replica) could not participate correctly.

---

## Key Concepts

### Leader vs. Follower Nodes

When a DP replica spans N nodes:
- **Leader node** (`node_rank % N == 0`): Hosts the EngineCore (scheduler, ZMQ,
  MessageQueue). Creates the `MultiprocExecutor` which spawns local GPU workers
  and the MessageQueue that remote workers connect to.
- **Follower node** (`node_rank % N != 0`): Hosts NO EngineCore. Only spawns a
  `MultiprocExecutor` that creates local GPU workers. These workers join the
  leader's torch.distributed group and connect to the leader's MessageQueue via
  the `_INNER_DP_WORLD` group.

### `data_parallel_size_local`

This field means "how many complete DP replicas does this node host?"

| Scenario | GPUs/node | TP×PP | dp_size_local |
|---|---|---|---|
| 4 replicas on 1 node (TP=2) | 8 | 2 | 4 |
| 2 replicas per node across 2 nodes (TP=2) | 8 | 2 | 2 |
| 1 replica per node (TP=8) | 8 | 8 | 1 |
| **Replica spans 2 nodes (TP=8, PP=2)** | **8** | **16** | **1 (leader) / 0 (follower)** |

---

## Architecture & Code Flow

### How the sbatch works

```bash
# All 8 nodes run the same script, differentiated by SLURM_NODEID
srun --ntasks-per-node=1 kimi_k2_multinode_dp.sbatch
```

- **Node 0** (`SLURM_NODEID=0`): `vllm serve $MODEL --port 8000 ...`
  → Head: API server + coordinator + EngineCore(dp=0) + 8 GPU workers
- **Node 1** (`SLURM_NODEID=1`): `vllm serve $MODEL --headless ...`
  → `dp_size_local=0`, `node_rank_within_dp=1` → executor-only (8 GPU workers)
- **Node 2** (`SLURM_NODEID=2`): `vllm serve $MODEL --headless ...`
  → `dp_size_local=1`, `node_rank_within_dp=0` → EngineCore(dp=1) + 8 GPU workers
- ... and so on.

### Worker Communication (torch.distributed)

All 64 GPU workers across 8 nodes join a **single global torch.distributed group**:

```
world_size = dp_size × TP × PP = 4 × 8 × 2 = 64
rank = dp_rank × (TP×PP) + local_rank_within_replica
```

This is possible because `init_distributed_environment()` **overrides** the
per-executor loopback `distributed_init_method` with `master_addr:master_port`
when `nnodes > 1`:

> **Reference:** `vllm/distributed/parallel_state.py:1214-1244`
>
> ```python
> if parallel_config.nnodes > 1:
>     ip = parallel_config.master_addr
>     port = parallel_config.master_port
>     distributed_init_method = get_distributed_init_method(ip, port)
> ```

This override is why we do NOT need to change `multiproc_executor.py:120-122`
(which uses `get_loopback_ip()`) — it gets replaced automatically for multi-node.

### MessageQueue Communication

Within each DP replica, the leader creates a `MessageQueue` that spans all 16
workers. Local workers (ranks 0-7) use shared memory. Remote workers (ranks 8-15
on the follower node) connect via TCP through the `_INNER_DP_WORLD` group:

> **Reference:** `vllm/v1/executor/multiproc_executor.py:499-530`
>
> ```python
> if vllm_config.parallel_config.nnodes_within_dp == 1:
>     self.rpc_broadcast_mq = MessageQueue.create_from_handle(...)  # shared memory
> else:
>     self.rpc_broadcast_mq = get_inner_dp_world_group().create_mq_broadcaster(...)  # TCP
> ```

The `_INNER_DP_WORLD` group is created automatically when `nnodes_within_dp > 1`:

> **Reference:** `vllm/distributed/parallel_state.py:1295-1311`

### The Pre-Existing Follower Path

The key discovery: `run_headless()` in `serve.py` **already had** a follower
executor-only path:

> **Reference:** `vllm/entrypoints/cli/serve.py:176-185`
>
> ```python
> if parallel_config.node_rank_within_dp > 0:
>     # Run headless workers (for multi-node PP/TP).
>     executor = MultiprocExecutor(vllm_config, monitor_workers=False)
>     executor.start_worker_monitor(inline=True)
>     return
> ```

This code was written for multi-node TP/PP without DP. It creates only an
executor (no EngineCore, no scheduler, no ZMQ) and blocks monitoring workers.
It was blocked from the multi-node DP replica case by the validation:
`if local_engine_count <= 0: raise ValueError(...)`.

---

## Changes Made

### 1. `vllm/engine/arg_utils.py` — Inference of `data_parallel_size_local`

**Before:** `max(local_world_size // world_size_within_dp, 1)` — always >= 1.

**After:** Branches on whether a replica fits on one node:

```python
if local_world_size >= world_size_within_dp:
    # Replica fits on one node. E.g., 8 GPUs, TP=2 → dp_local=4
    dp_size_local = local_world_size // world_size_within_dp
else:
    # Replica spans multiple nodes. E.g., 8 GPUs, TP=8×PP=2=16
    nodes_per_replica = world_size_within_dp // local_world_size  # = 2
    is_dp_leader = (node_rank % nodes_per_replica == 0)
    dp_size_local = 1 if is_dp_leader else 0
```

### 2. `vllm/config/parallel.py` — `nnodes_within_dp` property

**Before:** `nnodes // (dp_size // dp_size_local)` — divides by zero when
`dp_size_local=0`.

**After:** Compute from total parallelism config:

```python
total_gpus = data_parallel_size × world_size
gpus_per_node = total_gpus // nnodes
if world_size <= gpus_per_node:
    return 1  # replica fits on one node
return world_size // gpus_per_node
```

**Equivalence proof (for all existing cases where dp_size_local > 0):**

| Config | Old formula | New formula |
|---|---|---|
| DP=4, TP=2, 1 node | short-circuit → 1 | short-circuit → 1 |
| DP=4, TP=2, 2 nodes, dp_local=2 | 2//(4//2)=1 | total=8, gpn=4, 2≤4 → 1 |
| DP=4, TP=8, PP=2, 8 nodes, dp_local=1 | 8//(4//1)=2 | total=64, gpn=8, 16>8 → 2 |
| DP=4, TP=8, PP=2, 8 nodes, dp_local=0 | **div/0** | total=64, gpn=8, 16>8 → **2** |

### 3. `vllm/entrypoints/cli/serve.py` — Validation relaxation

**Before:** `if local_engine_count <= 0: raise ValueError`

**After:** Allow `dp_size_local=0` when `node_rank_within_dp > 0` (follower).
Still raise if a leader node somehow has `dp_size_local=0`.

### 4. `examples/online_serving/kimi_k2_multinode_dp.sbatch` — Added `--master-addr`

`--master-addr $HEAD_ADDR` is required for `torch.distributed.init_process_group`
across all 64 workers. Without it, `master_addr` defaults to `127.0.0.1` and the
override in `parallel_state.py:1214-1244` would use loopback, preventing
cross-node rendezvous.

---

## What We Did NOT Need to Change (and Why)

### `multiproc_executor.py:120` — `distributed_init_method` uses loopback

The loopback `distributed_init_method` is **overridden** in
`init_distributed_environment()` when `nnodes > 1`. Workers never actually use
the loopback address for cross-node communication. This was the biggest
"aha" moment in the investigation — it eliminated the riskiest change.

### `multiproc_executor.py:226` — `_post_init_executor` is a no-op

The `MultiprocExecutor._post_init_executor()` is `pass`. KV cache profiling
happens in the EngineCore, not the executor. So creating a standalone executor
on the follower doesn't trigger any unwanted coordination.

### `core.py` — EngineCore MoE vs non-MoE handling

For MoE models (like Kimi K2), `DPEngineCoreProc` is used and `dp_size` is
preserved. For non-MoE, `dp_size` is reset to 1 (replicas are independent).
Our changes only affect MoE models where `dp_size > 1` is preserved through
the worker initialization. Non-MoE multi-node replicas are a separate (unlikely)
edge case.

---

## Risks & Limitations

1. **Non-MoE multi-node replicas:** The EngineCore resets `dp_size=1` for
   non-MoE (core.py:993), which breaks `nnodes_within_dp` when replicas span
   nodes. This was broken before our changes too. In practice, dense models
   rarely need TP×PP > 8 WITH DP > 1.

2. **Timing dependency:** Follower workers must start before the leader's
   MessageQueue timeout. In practice, Slurm launches all nodes nearly
   simultaneously, but extreme delays could cause hangs.

3. **Port conflicts:** All replicas share the same `master_addr:master_port`
   for torch.distributed. This works because all 64 workers are in ONE global
   group. If the non-MoE `dp_size=1` reset were used with multi-node replicas,
   different replicas would conflict on the same port.

---

## Testing Notes

- **Existing tests unaffected:** The `test_internal_lb_dp.py` tests use
  `nnodes=1` (default) and explicit `--data-parallel-size-local`. Our inference
  code only runs when `nnodes > 1` AND `dp_size_local` is not explicitly set.

- **End-to-end validation** requires an actual multi-node Slurm cluster with
  64 GPUs. The logic can be unit-tested by mocking `node_rank`, `nnodes`, etc.

---

## File Reference Index

| File | Lines | What |
|---|---|---|
| `vllm/engine/arg_utils.py` | 1535-1560 | `dp_size_local` inference (CHANGED) |
| `vllm/config/parallel.py` | 466-478 | `nnodes_within_dp` property (CHANGED) |
| `vllm/entrypoints/cli/serve.py` | 154-161 | Validation for headless dp_size_local (CHANGED) |
| `vllm/entrypoints/cli/serve.py` | 176-185 | Pre-existing follower executor-only path |
| `vllm/distributed/parallel_state.py` | 1214-1244 | distributed_init_method override for nnodes > 1 |
| `vllm/distributed/parallel_state.py` | 1295-1311 | `_INNER_DP_WORLD` group creation |
| `vllm/v1/executor/multiproc_executor.py` | 119-122 | Loopback init_method (overridden, NOT changed) |
| `vllm/v1/executor/multiproc_executor.py` | 127-137 | MQ creation on leader with `connect_ip=master_addr` |
| `vllm/v1/executor/multiproc_executor.py` | 499-530 | Worker MQ init (shm vs inner_dp_world) |
| `vllm/v1/engine/core.py` | 985-996 | MoE preserves dp_size, non-MoE resets to 1 |
| `examples/online_serving/kimi_k2_multinode_dp.sbatch` | all | Example sbatch (CHANGED) |
