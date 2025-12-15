# char_rnn_model.py

import torch
import torch.nn as nn

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

        # Same loss logic
        loss = None
        if targets is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            )
        return logits, loss