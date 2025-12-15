# train.py (updated to fit the project's requirements and structure)
"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

See top-of-file comments in the original repo for usage examples.
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# default config values (these can be overridden by configurator.py)
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False  # if True, script exits right after the first eval
always_save_checkpoint = True  # if True, always save a checkpoint after each eval
init_from = 'scratch'  # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False
wandb_project = 'owt'
wandb_run_name = 'gpt2'
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8
batch_size = 12
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False
# adamw optimizer
learning_rate = 6e-4
max_iters = 600000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
# learning rate decay settings
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 600000
min_lr = 6e-5
# DDP settings
backend = 'nccl'
# system
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = True
# -----------------------------------------------------------------------------
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read())  # overrides from command line or config file
config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------

# DDP / device setup
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)

torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# data loader
data_dir = os.path.join('data', dataset)


def get_batch(split):
    # recreate memmap each call to avoid leaks
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


# init
iter_num = 0
best_val_loss = 1e9

# derive vocab size if possible
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta.get('vocab_size', None)
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout)
if init_from == 'scratch':
    print("Initializing a new model from scratch")
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    ckpt_path = os.path.join(out_dir, f"ckpt_{iter_num}.pt")
    print(f"Resuming training from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)

# possibly crop block size
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size
model.to(device)

# scaler and optimizer
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None

# compile if requested
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)

# wrap DDP
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helper to estimate loss (unchanged)
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# LR schedule
def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# wandb
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# -------------------------
# TRAINING TIMING / GPU METRICS (ADDED)
# -------------------------
# We'll measure wall-clock training time EXCLUDING eval + checkpoint writes.
last_eval_iter = 0
train_time_accum = 0.0  # seconds spent in actual training (forward/backward/opt) since last eval
# optional: keep a list of (iter, tokens_processed, seconds, tok_per_sec)
perf_history = []

# Zero peak memory stats at start if CUDA
if device_type == 'cuda':
    try:
        torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass

# -------------------------
# TRAINING LOOP
# -------------------------
X, Y = get_batch('train')  # prefetch first batch
t0 = time.time()
local_iter_num = 0
raw_model = model.module if ddp else model
running_mfu = -1.0

while True:
    # set LR
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # EVALUATION + CHECKPOINT (kept separate; we won't include checkpoint time in train time)
    if iter_num % eval_interval == 0 and master_process:
        # perform evaluation (this is NOT counted as training time)
        eval_start = time.time()
        losses = estimate_loss()
        eval_time = time.time() - eval_start

        # compute tokens processed since last eval (we only count real training tokens)
        iters_done = max(0, iter_num - last_eval_iter)
        tokens_done = tokens_per_iter * iters_done

        # compute throughput based on accumulated training time (exclude eval and checkpointing)
        sec_training = train_time_accum if train_time_accum > 0 else 0.0
        tps = tokens_done / sec_training if sec_training > 0 else 0.0

        # GPU memory stats (peak during training/eval since last reset)
        gpu_peak_alloc = None
        gpu_peak_reserved = None
        gpu_current_alloc = None
        if device_type == 'cuda':
            try:
                # If users run on 'cuda:0' this still works
                gpu_peak_alloc = torch.cuda.max_memory_allocated(device=device) / (1024 ** 3)  # in GB
                gpu_peak_reserved = torch.cuda.max_memory_reserved(device=device) / (1024 ** 3)
                gpu_current_alloc = torch.cuda.memory_allocated(device=device) / (1024 ** 3)
                # reset so next interval measures fresh peak
                torch.cuda.reset_peak_memory_stats(device=device)
            except Exception:
                gpu_peak_alloc = gpu_peak_reserved = gpu_current_alloc = None

        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        print(f"  eval_time={eval_time:.3f}s, train_time_since_last_eval={sec_training:.3f}s, tokens={tokens_done:,}, tok/s={tps:,.1f}")
        if device_type == 'cuda':
            print(f"  GPU peak alloc (GB): {gpu_peak_alloc}, peak reserved (GB): {gpu_peak_reserved}, current alloc (GB): {gpu_current_alloc}")

        # optionally log to wandb
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu * 100,
                "tokens_since_last_eval": tokens_done,
                "train_time_since_last_eval": sec_training,
                "tokens_per_sec": tps,
                "gpu_peak_alloc_gb": gpu_peak_alloc,
                "gpu_peak_reserved_gb": gpu_peak_reserved,
                "gpu_current_alloc_gb": gpu_current_alloc,
            })

        # save checkpoint if best or always_save_checkpoint
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            # save checkpoint with iter in the filename (we want ckpt_0.pt as well)
            checkpoint = {
                'model': raw_model.state_dict(),
                'model_args': model_args,
                'iter_num': iter_num,
                'best_val_loss': best_val_loss
            }
            ckpt_name = f"ckpt_{iter_num}.pt"
            print(f"saving checkpoint to {out_dir}/{ckpt_name}")
            torch.save(checkpoint, os.path.join(out_dir, ckpt_name))

        # record performance history & reset training accumulators
        perf_history.append({
            'iter': iter_num,
            'tokens': tokens_done,
            'train_seconds': sec_training,
            'tokens_per_sec': tps,
            'val_loss': losses['val'],
        })
        last_eval_iter = iter_num
        train_time_accum = 0.0  # reset training-time accumulator (we will start fresh measuring from now)

    if iter_num == 0 and eval_only:
        break

    # ---------- TRAINING STEP (we WILL measure this time) ----------
    # we'll time only the forward/backward/optimizer.step part and accumulate it
    t_train_block_start = time.time()

    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps
        # prefetch next batch asynchronously
        X, Y = get_batch('train')
        # backward (with grad scaling)
        scaler.scale(loss).backward()

    # gradient clipping + optimizer step
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # measure elapsed training time for this block and accumulate
    t_train_block_end = time.time()
    elapsed_training = t_train_block_end - t_train_block_start
    train_time_accum += elapsed_training

    # timing & logging (keeps previous behavior)
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:
            try:
                mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            except Exception:
                mfu = 0.0
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")

    iter_num += 1
    local_iter_num += 1

    # termination
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()