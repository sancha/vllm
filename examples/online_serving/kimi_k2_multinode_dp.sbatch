#!/bin/bash
#SBATCH --job-name=kimi-k2-dp4
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --output=kimi-k2-%j.log

# Kimi K2 with TP=8, PP=2, DP=4 across 8 nodes (64 GPUs total)
#
# Topology:
#   Each DP replica = TP(8) x PP(2) = 16 GPUs = 2 nodes
#   4 replicas x 2 nodes/replica = 8 nodes
#
#   Nodes 0,1 -> DP replica 0  (node 0 = leader engine, node 1 = workers only)
#   Nodes 2,3 -> DP replica 1  (node 2 = leader engine, node 3 = workers only)
#   Nodes 4,5 -> DP replica 2  (node 4 = leader engine, node 5 = workers only)
#   Nodes 6,7 -> DP replica 3  (node 6 = leader engine, node 7 = workers only)
#
#   Node 0 additionally runs the API server, coordinator, and internal LB.

MODEL="moonshotai/Kimi-K2-Instruct"
HEAD_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
DP_RPC_PORT=13345
API_PORT=8000

TP=8
PP=2
DP=4

COMMON_ARGS="
    --tensor-parallel-size $TP
    --pipeline-parallel-size $PP
    --data-parallel-size $DP
    --nnodes 8
    --node-rank $SLURM_NODEID
    --data-parallel-address $HEAD_ADDR
    --data-parallel-rpc-port $DP_RPC_PORT
    --trust-remote-code
    --enable-expert-parallel
"

# ---- Auto-detect role from node rank ----
#
# Today's vLLM requires this to be split into head + headless.
# With the proposed changes, vLLM would auto-infer from --node-rank:
#
#   nodes_per_replica = ceil(TP*PP / gpus_per_node) = ceil(16/8) = 2
#   is_dp_leader      = (node_rank % nodes_per_replica) == 0
#   dp_rank            = node_rank // nodes_per_replica
#   data_parallel_size_local = 1 if is_dp_leader else 0
#
# Node 0 is the head (API server + coordinator).
# Even-numbered nodes (0,2,4,6) are DP leaders  -> run EngineCore + local workers.
# Odd-numbered nodes  (1,3,5,7) are DP followers -> run workers only (no EngineCore).

if [ "$SLURM_NODEID" -eq 0 ]; then
    # Head node: API server + coordinator + DP leader for replica 0
    vllm serve $MODEL $COMMON_ARGS \
        --port $API_PORT
else
    # All other nodes: headless
    vllm serve $MODEL $COMMON_ARGS \
        --headless
fi
