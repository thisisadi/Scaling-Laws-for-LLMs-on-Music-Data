
# =====================
#  ~1M PARAM MODEL (1 epoch on 120M tokens)
# =====================

out_dir = "out-music-char-1M"
eval_interval = 250
log_interval = 10

eval_iters = 50
eval_only = False

always_save_checkpoint = True
init_from = "scratch"

# =====================
# Data
# =====================
dataset = "../data/my_music_subset" # subset of ~ 1 billion tokens
gradient_accumulation_steps = 1
batch_size = 32                # ~16k tokens/iter
block_size = 512

# =====================
# Model Architecture
# =====================
n_layer = 4
n_head = 4
n_embd = 144
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

# tokens per iteration = batch_size * block_size
tokens_per_iter = batch_size * block_size      # 32 * 512 = 16384

# number of iterations for 1 epoch:
max_iters = 120_000_000 // tokens_per_iter     # ~7324 iters

# =====================
# System
# =====================
device = "cuda"
dtype = "bfloat16"
compile = False
