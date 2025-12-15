import os
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

input_dir = os.path.join(BASE_DIR, "data/abc_raw")
output_dir = os.path.join(BASE_DIR, "data/abc_clean")
log_file = os.path.join(BASE_DIR, "data/logs/cleaning_log.txt")

os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.dirname(log_file), exist_ok=True)


def is_valid_abc(text):
    """Heuristics to check if an ABC file is usable."""
    if len(text.strip()) < 200:  # too short
        return False
    if "K:" not in text:  # missing key signature
        return False
    if "M:" not in text:  # missing meter
        return False
    return True


def clean_file(input_path, output_path):
    try:
        with open(input_path, "r", errors="ignore") as f:
            text = f.read()

        # Basic cleaning
        text = text.replace("\r\n", "\n").strip() + "\n"

        if not is_valid_abc(text):
            return False, input_path

        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w") as f:
            f.write(text)

        return True, input_path

    except Exception as e:
        return False, f"{input_path} --- {str(e)}"


def main():
    all_files = []

    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.endswith(".abc"):
                abs_path = os.path.join(root, f)

                # Mirror path in abc_clean/
                relative = os.path.relpath(abs_path, input_dir)
                out_path = os.path.join(output_dir, relative)

                all_files.append((abs_path, out_path))

    print(f"Found {len(all_files)} ABC files to clean")

    failures = []

    for inp, outp in tqdm(all_files):
        ok, msg = clean_file(inp, outp)
        if not ok:
            failures.append(msg)

    with open(log_file, "w") as f:
        for line in failures:
            f.write(line + "\n")

    print(f"Cleaned: {len(all_files) - len(failures)}, Failed: {len(failures)}")


if __name__ == "__main__":
    main()