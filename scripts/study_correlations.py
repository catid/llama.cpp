import os
import threading
import struct
import shutil
import re
import subprocess

app_path = "/home/catid/sources/llama.cpp"

# Be careful here about memory usage.  Each thread needs over 400 MB (Mistral 7B) depending on the size of the correlation matrices.
# So 24 threads would take 10 GB of RAM.  This seems reasonable but you may need to set it lower.
nthreads = 24

def study_file(filename, _):
    print(f"Studying {filename}")

    try:
        # Construct the command
        command = f"{app_path}/build/bin/study_correlations {filename}"

        # Run the command
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    except subprocess.CalledProcessError as e:
        # Handle the case where the command failed
        print(f"Command failed with exit code {e.returncode}")
        print("Error message:", e.stderr.decode())
        raise ValueError("Summation failed")

def process_files_in_threads(out_files):
    threads = []
    for i in range(0, len(out_files)):
        thread = threading.Thread(target=study_file, args=(out_files[i], None))
        threads.append(thread)
        thread.start()

        # If we have 24 active threads, wait for them to complete before starting more
        if len(threads) >= nthreads:
            for t in threads:
                t.join()
            threads = []  # Reset the list for the next batch of threads

    # Wait for any remaining threads to complete
    for t in threads:
        t.join()

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def main():
    outputs_dir = './outputs/'

    if not os.path.exists(outputs_dir):
        print(f"Folder is missing: {outputs_dir}.  Please follow the instructions in CORRELATIONS.md")
        return

    out_files = [os.path.join(outputs_dir, f) for f in os.listdir(outputs_dir) if f.startswith('correlations_block_')]
    out_files.sort(key=natural_sort_key)

    if len(out_files) == 0:
        print(f"No data found in the {outputs_dir} folder.  Please follow the instructions in CORRELATIONS.md")
        return

    process_files_in_threads(out_files)

if __name__ == "__main__":
    main()
