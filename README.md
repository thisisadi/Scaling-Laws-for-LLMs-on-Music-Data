# Scaling Laws for Language Models on Music Data

This repository explores scaling laws for character-level language models trained on symbolic music (ABC notation). We compare Transformer-based models (nanoGPT) and LSTM-based RNNs across multiple parameter scales to study how validation loss and generation quality improve with model size.

The project includes full training, evaluation, scaling analysis, and music generation pipelines.

# Google Drive Structure

```
├── nanoGPT/
│   ├── config/                  # Transformer configs (1M–100M parameters)
│   ├── train.py                 # Modified nanoGPT training script
│   ├── model.py                 # nanoGPT model definition
│   ├── generated_outputs/       # Generated ABC + MIDI samples
│   └── out-music-char-*         # Transformer checkpoints
│
├── RNN/
│   ├── config/                  # RNN (LSTM) configs (1M–50M parameters)
│   ├── char_rnn_model.py        # Custom LSTM character model
│   ├── train_lstm.py            # RNN training script
│   └── out-rnn-char-*           # RNN checkpoints
│
├── data/
│   └── my_music_subset/         # Tokenized dataset (train/val/test/meta)
```

# Dataset Preparation
- All music preprocessing (MIDI → ABC → tokenization) was performed locally.
- The tokenized dataset was uploaded to Google Drive for training.

The dataset directory must contain:

- train.bin
- val.bin
- test.bin
- meta.pkl
- vocab.json

For efficient training, a 120M-token subset was created from the original ~1B-token dataset, while keeping validation and test splits unchanged.

#Environment Setup (Google Colab)

```
from google.colab import drive
drive.mount('/content/drive')

# Clone nanoGPT only once (first run)
# !git clone https://github.com/karpathy/nanoGPT.git

%cd /content/drive/MyDrive/nanoGPT

Install MIDI conversion tools (for generation):

!apt-get install -y abcmidi
```

# Transformer Training (nanoGPT)

Five Transformer models were trained with increasing parameter counts:

```
Config File	Parameters
music_char_1M.py	~1M
music_char_5M.py	~5M
music_char_20M.py	~20M
music_char_50M.py	~50M
music_char_100M.py	~100M

Train a model using:

python train.py config/music_char_1M.py
```

Transformer Evaluation & Scaling Analysis
- All saved checkpoints are evaluated on the validation set.
- Validation loss vs. training iteration plots are generated.
- A power-law scaling curve is fit between model size and validation loss.

The evaluation notebook includes:
- Checkpoint evaluation
- Power-law fitting
- Visualization of scaling behavior

# RNN (LSTM) Training

Four LSTM-based character models were trained:

Config File	Parameters
- rnn_char_1M.py	~1M
- rnn_char_5M.py	~5M
- rnn_char_20M.py	~20M
- rnn_char_50M.py	~50M

Train RNN models:

```
cd /content/drive/MyDrive/RNN
python train_lstm.py config/rnn_char_1M.py
```

# RNN Evaluation & Scaling
- Validation loss is computed for each checkpoint.
- Loss vs. iteration plots are generated.
- A log-linear scaling law is fit between RNN size and validation loss.
- RNN scaling behavior is compared directly with Transformers.

# Final Test Evaluation
- The 100M-parameter Transformer achieved the lowest validation and test loss.
- Final test-set evaluation reports:
- Test loss
- Perplexity

This model is selected for music generation.

# ABC Music Generation

Using the best Transformer model:
- Unconditional generation
- Conditional generation (with ABC prefixes)
- Automatic header insertion if missing
- Conversion to MIDI using abc2midi
- Generation statistics reported:
- Syntax validity
- MIDI-convertibility rate

Generated outputs are saved under:

```
nanoGPT/generated_outputs/
```