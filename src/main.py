import argparse
import time
import torch
import torch.optim as optim
from config.config import (
    BATCH_SIZE,
    HIDDEN_DIM,
    TOTAL_LAYERS,
    STEPS,
    CHUNKS,
    RANDOM_SEED
)
from model import ShardedMLP
from profiler import PipelineProfiler
from communication import init_distributed, PipelineCommunication
from schedule import (
    naive_pipeline_step,
    gpipe_pipeline_step,
    onef_oneb_pipeline_step,
    microbatch_sizes,
)

parser = argparse.ArgumentParser(description="Train a sharded MLP with pipeline parallelism")
parser.add_argument("--schedule", choices=("naive", "gpipe", "1f1b"), default="1f1b")
parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
parser.add_argument("--chunks", type=int, default=CHUNKS)
parser.add_argument("--steps", type=int, default=STEPS)
args = parser.parse_args()
if args.batch_size < 1 or args.steps < 1:
    parser.error("batch size and steps must be positive")
if args.schedule != "naive":
    try:
        microbatch_sizes(args.batch_size, args.chunks)
    except ValueError as error:
        parser.error(str(error))

# 1. Distributed Environment
rank, world_size, device = init_distributed()
communication = PipelineCommunication(rank, world_size)
profiler = PipelineProfiler(rank)

torch.manual_seed(RANDOM_SEED)

# Each rank needs to "skip" the random numbers used by previous ranks

# The skip count is:
# skip = rank * (layers_per_rank * rand_calls_per_layer)

# where:
# rand_calls_per_layer = 2  (weight + bias)

# The effect:
# Rank 0 skips 0 → generates its layers → RNG pointer moves forward naturally.
# Rank 1 skips exactly the number of random values Rank 0 consumed 
# → lands at the correct global RNG position 
# → generates its own layers as if it were continuing the same single-GPU model.
# Rank 2 skips exactly the number Rank 0 + Rank 1 consumed → continues from there.

for i in range(
    rank * (TOTAL_LAYERS // world_size) * 2
):  # 2 params per layer (weight, bias)
    torch.randn(1)  # Consume RNG state


if rank == 0:
    print(f"--- Starting Micro PP on {world_size} Processes (GPU/CPU) ---")

# 2. Initialize the Sharded Model
model = ShardedMLP(
    HIDDEN_DIM,
    TOTAL_LAYERS,
    rank,
    world_size
).to(device)

# 3. Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. Only Rank 0 loads the data.
if rank == 0:
    fixed_input = torch.randn(args.batch_size, HIDDEN_DIM).to(device)
else:
    fixed_input = args.batch_size

# 5. Only the Last Rank needs the targets to calc loss.
if rank == world_size - 1:
    # We want the model to learn to classify 
    # these random vectors into class '0' or '1'
    fixed_target = torch.randint(0, 2, (args.batch_size,)).to(device)
else:
    fixed_target = torch.empty(0).to(device)

model.train()

loss = None
start_time = time.time()

for step in range(args.steps):
    profiler.start_stage("step")
    optimizer.zero_grad()
    if args.schedule == "naive":
        loss = naive_pipeline_step(model, communication, fixed_input, fixed_target,
                                   HIDDEN_DIM, device, profiler)
    else:
        pipeline_step = (gpipe_pipeline_step if args.schedule == "gpipe"
                         else onef_oneb_pipeline_step)
        loss = pipeline_step(model, communication, fixed_input, fixed_target,
                             HIDDEN_DIM, args.chunks, device, profiler)
    optimizer.step()
    profiler.end_stage("step")
    if rank == world_size - 1 and step % 5 == 0 and loss is not None:
        print(f"Step {step:02d} | Loss: {loss.item():.6f}")

# Clean up
duration = time.time() - start_time
if rank == 0:
    print("--- Training Complete ---")
    print(f"Time: {duration:.3f}s")
if rank == world_size - 1 and loss is not None:
    print(f"Final Loss: {loss.item():.6f}")
profiler.print_summary()
torch.distributed.destroy_process_group()


# --- Starting Micro PP on 4 Processes (GPU/CPU) ---
# Step 00 | Loss: 0.698048
# Step 05 | Loss: 0.688032
# Step 10 | Loss: 0.685432
# Step 15 | Loss: 0.684712
# Step 20 | Loss: 0.659281
# Step 25 | Loss: 0.408974
# Step 30 | Loss: 0.358847
# Step 35 | Loss: 0.313759
# Step 40 | Loss: 0.287643
# Step 45 | Loss: 0.265489
# Final Loss: 0.214853

# --- Pipeline Profiler (Rank 3) ---
# Steps: 50 | Avg step time: 0.0054s
# Compute: 0.0426s (15.8%)
# Comm:    0.1830s (68.0%)
# Bubbles: 0.0434s (16.1%)
# Total:   0.2689s
#   compute_backward    : 0.0005s avg (50 calls)
#   compute_forward     : 0.0003s avg (50 calls)
#   recv_forward        : 0.0036s avg (50 calls)
#   send_backward       : 0.0001s avg (50 calls)

# --- Pipeline Profiler (Rank 2) ---
# Steps: 50 | Avg step time: 0.0053s
# Compute: 0.0389s (14.8%)
# Comm:    0.1820s (69.0%)
# Bubbles: 0.0428s (16.2%)
# Total:   0.2637s
#   compute_backward    : 0.0005s avg (50 calls)
#   compute_forward     : 0.0003s avg (50 calls)
#   recv_backward       : 0.0012s avg (50 calls)
#   recv_forward        : 0.0023s avg (50 calls)
#   send_backward       : 0.0001s avg (50 calls)
#   send_forward        : 0.0001s avg (50 calls)

# --- Pipeline Profiler (Rank 1) ---
# Steps: 50 | Avg step time: 0.0054s
# Compute: 0.0355s (13.1%)
# Comm:    0.1860s (68.7%)
# Bubbles: 0.0492s (18.2%)
# Total:   0.2707s
#   compute_backward    : 0.0005s avg (50 calls)
#   compute_forward     : 0.0002s avg (50 calls)
#   recv_backward       : 0.0025s avg (50 calls)
#   recv_forward        : 0.0011s avg (50 calls)
#   send_backward       : 0.0001s avg (50 calls)
#   send_forward        : 0.0001s avg (50 calls)
# --- Training Complete ---
# Time: 0.266s

# --- Pipeline Profiler (Rank 0) ---
# Steps: 50 | Avg step time: 0.0053s
# Compute: 0.0396s (14.9%)
# Comm:    0.1857s (70.0%)
# Bubbles: 0.0401s (15.1%)
# Total:   0.2654s
#   compute_backward    : 0.0006s avg (50 calls)
#   compute_forward     : 0.0002s avg (50 calls)
#   recv_backward       : 0.0036s avg (50 calls)
#   send_forward        : 0.0001s avg (50 calls)
