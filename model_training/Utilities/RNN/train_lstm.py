import os
import math
import time
import pickle
import numpy as np
import torch
import torch.nn as nn

# ------------------------
# Load config
# ------------------------
config_file = None
import sys
if len(sys.argv) > 1:
    config_file = sys.argv[1]
if config_file:
    exec(open(config_file).read())

# Required by the assignment
assert block_size == 512
assert batch_size == 32

device = "cuda" if torch.cuda.is_available() else "cpu"
device_type = "cuda" if device == "cuda" else "cpu"

# ------------------------
# LSTM Language Model
# ------------------------

class CharRNN(nn.Module):
    def __init__(self, vocab_size, n_embd, n_layer, dropout):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, n_embd)

        self.lstm = nn.LSTM(
            input_size=n_embd,
            hidden_size=n_embd,
            num_layers=n_layer,
            batch_first=True,
            dropout=dropout
        )

        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, x, targets=None):
        emb = self.embed(x)
        out, _ = self.lstm(emb)
        logits = self.lm_head(out)

        loss = None
        if targets is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )
        return logits, loss


# ------------------------
# Dataset loading
# ------------------------
data_dir = "../data/my_music_subset"

train_data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
val_data   = np.memmap(os.path.join(data_dir, "val.bin"),   dtype=np.uint16, mode="r")

with open(os.path.join(data_dir, "meta.pkl"), "rb") as f:
    meta = pickle.load(f)
vocab_size = len(meta["itos"])

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# ------------------------
# Initialize RNN
# ------------------------
model_args = dict(
    n_embd = n_embd,
    n_layer = n_layer,
    dropout = dropout,
    vocab_size = vocab_size,
    block_size = block_size
)

model = CharRNN(
    vocab_size=vocab_size,
    n_embd=n_embd,
    n_layer=n_layer,
    dropout=dropout
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Print total params
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")

# ------------------------
# Training Loop
# ------------------------
tokens_per_iter = batch_size * block_size
print("tokens_per_iter =", tokens_per_iter)
print("max_iters:", max_iters)

os.makedirs(out_dir, exist_ok=True)

def estimate_loss():
    was_training = model.training
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(eval_iters):
            X, Y = get_batch("val")
            _, loss = model(X, Y)
            losses.append(loss.item())
    if was_training:
        model.train()
    return np.mean(losses)

iter_num = 0
last_eval_iter = 0
train_time_accum = 0.0

# Reset GPU peak stats
if device_type == "cuda":
    torch.cuda.reset_peak_memory_stats()

print("\n==== Starting Training ====\n")

while iter_num <= max_iters:
    model.train()

    # ------------------------
    # EVALUATE + CHECKPOINT
    # ------------------------
    if iter_num % eval_interval == 0:
        eval_start = time.time()
        val_loss = estimate_loss()
        eval_time = time.time() - eval_start

        # compute tokens processed since last eval
        iters_done = max(0, iter_num - last_eval_iter)
        tokens_done = iters_done * tokens_per_iter

        # compute throughput based only on training time (not eval or checkpoint)
        tps = tokens_done / train_time_accum if train_time_accum > 0 else 0.0

        # GPU memory
        gpu_peak_alloc = gpu_peak_reserved = gpu_current_alloc = None
        if device_type == "cuda":
            gpu_peak_alloc = torch.cuda.max_memory_allocated() / (1024**3)
            gpu_peak_reserved = torch.cuda.max_memory_reserved() / (1024**3)
            gpu_current_alloc = torch.cuda.memory_allocated() / (1024**3)
            torch.cuda.reset_peak_memory_stats()

        print(f"\nstep {iter_num}: val_loss={val_loss:.4f}")
        print(f"  eval_time={eval_time:.3f}s, train_time_since_last_eval={train_time_accum:.3f}s")
        print(f"  tokens={tokens_done}, tok/s={tps:,.1f}")
        if device_type == "cuda":
            print(f"  GPU peak alloc (GB): {gpu_peak_alloc:.3f}, peak reserved (GB): {gpu_peak_reserved:.3f}, current alloc (GB): {gpu_current_alloc:.3f}")

        # Save checkpoint
        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model_args": model_args,
            "iter_num": iter_num,
            "val_loss": val_loss,
        }
        torch.save(ckpt, os.path.join(out_dir, f"ckpt_{iter_num}.pt"))

        last_eval_iter = iter_num
        train_time_accum = 0.0

    # ------------------------
    # TRAINING STEP
    # ------------------------
    t_train_start = time.time()

    X, Y = get_batch("train")
    _, loss = model(X, Y)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    train_time_accum += (time.time() - t_train_start)

    if iter_num % log_interval == 0:
        print(f"Iter {iter_num}: train_loss={loss.item():.4f}")

    iter_num += 1