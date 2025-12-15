# ==========================
#   ~1M PARAM RNN MODEL
# ==========================

out_dir = "out-rnn-char-1M"

# dataset
dataset = "../data/my_music_subset"
batch_size = 32
block_size = 512
gradient_accumulation_steps = 1

# ==========================
# Architecture (≈1.1M params)
# ==========================
n_layer = 2          # 2 LSTM layers
n_embd = 256         # embedding + hidden size
dropout = 0.1

# ==========================
# Optimization
# ==========================
learning_rate = 3e-4
eval_interval = 250
eval_iters = 50
log_interval = 10
always_save_checkpoint = True

# ==========================
# Training schedule
# ==========================
tokens_per_iter = batch_size * block_size
max_iters = 120_000_000 // tokens_per_iter  # ~7324 iters

# ==========================
# System
# ==========================
device = "cuda"
dtype = "bfloat16"
compile = False