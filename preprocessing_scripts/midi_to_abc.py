import os
import subprocess
from tqdm import tqdm
from multiprocessing import Pool

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Input/output paths
input_dir = os.path.join(BASE_DIR, "data/midi_raw/clean_midi")
output_dir = os.path.join(BASE_DIR, "data/abc_raw")
log_file = os.path.join(BASE_DIR, "data/logs/conversion_log.txt")

os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.dirname(log_file), exist_ok=True)


def convert_single_midi(path):
    """Convert one MIDI file to ABC notation using midi2abc."""
    # Mirror directory structure under abc_raw
    relative_path = os.path.relpath(path, os.path.join(BASE_DIR, "data/midi_raw"))
    out_path = os.path.join(output_dir, relative_path).replace(".mid", ".abc")

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    try:
        # Run midi2abc
        result = subprocess.run(
            ["midi2abc", path],
            capture_output=True,
            text=True,
            timeout=10  # prevent hangs
        )

        if result.returncode != 0:
            return False, f"{path} --- midi2abc error: {result.stderr}"

        # Save output to file
        with open(out_path, "w") as f:
            f.write(result.stdout)

        return True, path

    except Exception as e:
        return False, f"{path} --- {str(e)}"


def main():
    midi_files = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith(".mid") or f.lower().endswith(".midi"):
                midi_files.append(os.path.join(root, f))

    print(f"Found {len(midi_files)} MIDI files")

    # Parallel processing
    with Pool() as p:
        results = list(tqdm(p.imap(convert_single_midi, midi_files), total=len(midi_files)))

    # Log failures
    with open(log_file, "w") as f:
        for success, msg in results:
            if not success:
                f.write(msg + "\n")

    success_count = sum(1 for s, _ in results if s)
    print(f"Success: {success_count}, Failures: {len(results) - success_count}")


if __name__ == "__main__":
    main()