
out_dir = "out-music-char-5M"
eval_interval = 250
log_interval = 10

eval_iters = 50
eval_only = False
always_save_checkpoint = True
init_from = "scratch"

# =====================
# Data
# =====================
dataset = "../data/my_music_subset"
gradient_accumulation_steps = 1
batch_size = 32
block_size = 512     # identical across all models

# =====================
# Model Architecture (≈5M params)
# =====================
n_layer = 6
n_head = 8
n_embd = 256
dropout = 0.1

# =====================
# Learning Rate
# =====================
learning_rate = 3e-4
warmup_iters = 2000
lr_decay_iters = 200000
min_lr = 1e-5

# =====================
# Train exactly 1 epoch
# =====================
tokens_per_iter = batch_size * block_size
max_iters = 120_000_000 // tokens_per_iter   # ≈ 7324 iterations

# =====================
# System
# =====================
device = "cuda"
dtype = "bfloat16"
compile = False
