import os
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

input_dir = os.path.join(BASE_DIR, "data/abc_clean")
corpus_dir = os.path.join(BASE_DIR, "data/corpus")
corpus_path = os.path.join(corpus_dir, "corpus.txt")

os.makedirs(corpus_dir, exist_ok=True)


def main():
    abc_files = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.endswith(".abc"):
                abc_files.append(os.path.join(root, f))

    print(f"Merging {len(abc_files)} ABC files into corpus.txt...")

    with open(corpus_path, "w") as out:
        for path in tqdm(abc_files):
            with open(path, "r") as f:
                text = f.read().strip()

                out.write("<SONG_START>\n")
                out.write(text)
                out.write("\n<SONG_END>\n\n")

    # Summary stats
    size = os.path.getsize(corpus_path)
    print(f"Done! Corpus size: {size / 1e6:.2f} MB")

    # count characters/tokens
    with open(corpus_path, "r") as f:
        data = f.read()
        print(f"Total characters/tokens: {len(data):,}")
        print(f"Approx songs: {data.count('<SONG_START>'):,}")


if __name__ == "__main__":
    main()