import os
import json
import pickle
import numpy as np
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Input corpus
corpus_path = os.path.join(BASE_DIR, "data/corpus/corpus.txt")

# Output directory
out_dir = os.path.join(BASE_DIR, "data/char")
os.makedirs(out_dir, exist_ok=True)


def main():
    # Read entire corpus
    with open(corpus_path, "r") as f:
        data = f.read()

    print(f"Loaded corpus with {len(data):,} characters")

    # Build vocabulary (sorted for determinism)
    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    print(f"Vocab size: {vocab_size}")

    # Mapping char -> int
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    # Encode the entire dataset
    encoded = np.array([stoi[c] for c in tqdm(data)], dtype=np.uint16)

    # Train/val/test split: 98% / 1% / 1%
    n = len(encoded)
    n_train = int(0.98 * n)
    n_val = int(0.01 * n)
    n_test = n - n_train - n_val

    train_ids = encoded[:n_train]
    val_ids = encoded[n_train:n_train+n_val]
    test_ids = encoded[n_train+n_val:]

    print(f"Train tokens: {len(train_ids):,}")
    print(f"Val tokens:   {len(val_ids):,}")
    print(f"Test tokens:  {len(test_ids):,}")

    # Save binary .bin files
    train_ids.tofile(os.path.join(out_dir, "train.bin"))
    val_ids.tofile(os.path.join(out_dir, "val.bin"))
    test_ids.tofile(os.path.join(out_dir, "test.bin"))

    # Save metadata
    meta = {
        "vocab_size": vocab_size,
        "itos": itos,
        "stoi": stoi,
    }

    with open(os.path.join(out_dir, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    # Save vocab.json (useful for documentation)
    with open(os.path.join(out_dir, "vocab.json"), "w") as f:
        json.dump(chars, f, indent=4)

    print("Character-level dataset prepared successfully!")


if __name__ == "__main__":
    main()