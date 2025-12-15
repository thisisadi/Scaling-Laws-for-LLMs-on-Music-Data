out_dir = "out-rnn-char-5M"

dataset = "../data/my_music_subset"
batch_size = 32
block_size = 512
gradient_accumulation_steps = 1

# ~5M PARAM RNN
n_layer = 3
n_embd = 448      # gives ~4.9M total params
dropout = 0.1

learning_rate = 3e-4
eval_interval = 250
eval_iters = 50
log_interval = 10
always_save_checkpoint = True

tokens_per_iter = batch_size * block_size
max_iters = 120_000_000 // tokens_per_iter

device = "cuda"
dtype = "bfloat16"
compile = False