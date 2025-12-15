out_dir = "out-rnn-char-20M"

# --------------------
# Data
# --------------------
dataset = "../data/my_music_subset"
batch_size = 32
block_size = 512
gradient_accumulation_steps = 1

# --------------------
# Model Architecture (~19.3M params)
# --------------------
n_layer = 4
n_embd = 768
dropout = 0.1

# --------------------
# Optimization
# --------------------
learning_rate = 3e-4
eval_interval = 250
eval_iters = 50
log_interval = 10
always_save_checkpoint = True

# Train exactly 1 epoch on 120M tokens
tokens_per_iter = batch_size * block_size
max_iters = 120_000_000 // tokens_per_iter   # ≈ 7324

# --------------------
# System
# --------------------
device = "cuda"
dtype = "bfloat16"
compile = False