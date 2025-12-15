
# =====================
#  ~50M PARAM MODEL
#  (trained for exactly 1 epoch on 120M tokens)
# =====================

out_dir = "out-music-char-50M"
eval_interval = 250
log_interval = 10

eval_iters = 50
eval_only = False
always_save_checkpoint = True
init_from = "scratch"

# =====================
# Data (consistent across models)
# =====================
dataset = "../data/my_music_subset"
gradient_accumulation_steps = 1
batch_size = 32              # 32 * 512 = 16384 tokens per iteration
block_size = 512             # must remain identical across all models

# =====================
# Model Architecture (~50M params)
# =====================
n_layer = 12
n_head = 12
n_embd = 588      # 588/12 = 49 → valid
dropout = 0.1
bias = False

# =====================
# Learning Rate (consistent across models)
# =====================
learning_rate = 3e-4
warmup_iters = 2000
lr_decay_iters = 200000
min_lr = 1e-5

# =====================
# Train exactly 1 epoch
# =====================
tokens_per_iter = batch_size * block_size          # 16,384 tokens/iter
max_iters = 120_000_000 // tokens_per_iter         # ≈ 7324 iterations

# =====================
# System
# =====================
device = "cuda"
dtype = "bfloat16"
compile = False
