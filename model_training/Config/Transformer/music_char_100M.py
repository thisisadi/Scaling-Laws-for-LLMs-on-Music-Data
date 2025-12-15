
out_dir = "out-music-char-100M"
eval_interval = 250
log_interval = 10

eval_iters = 50
eval_only = False
always_save_checkpoint = True
init_from = "scratch"

# =====================
# Data (consistent across all models)
# =====================
dataset = "../data/my_music_subset"
gradient_accumulation_steps = 1
batch_size = 32
block_size = 512      # must remain identical

# =====================
# Model Architecture (~102M params)
# =====================
n_layer = 16          # ↑ depth increases params strongly
n_head  = 12          # must divide n_embd
n_embd  = 768         # GPT-2 small width
dropout = 0.1
bias = False

# =====================
# Learning Rate (same for all models)
# =====================
learning_rate = 3e-4
warmup_iters = 2000
lr_decay_iters = 200000
min_lr = 1e-5

# =====================
# 1-epoch training schedule
# =====================
tokens_per_iter = batch_size * block_size  # 16,384
max_iters = 120_000_000 // tokens_per_iter # ≈ 7324

# =====================
# System
# =====================
device = "cuda"
dtype = "bfloat16"
compile = False
