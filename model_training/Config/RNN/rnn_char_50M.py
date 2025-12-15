# =====================
# ~50M PARAM RNN MODEL
# =====================

out_dir = "out-rnn-char-50M"

# Data
dataset = "../data/my_music_subset"
batch_size = 32
block_size = 512
gradient_accumulation_steps = 1

# ~50M PARAM RNN
n_layer = 3
n_embd = 1400       # ~47.8M params
dropout = 0.1

# Optimization & logging
learning_rate = 3e-4
eval_interval = 250
eval_iters = 50
log_interval = 10
always_save_checkpoint = True

# Train exactly 1 epoch
tokens_per_iter = batch_size * block_size         # 16384
max_iters = 120_000_000 // tokens_per_iter        # ~7324

# System
device = "cuda"
dtype = "bfloat16"
compile = False