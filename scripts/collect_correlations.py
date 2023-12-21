import os
import threading
import struct
import shutil
import re
import subprocess

app_path = "/home/catid/sources/llama.cpp"
nthreads = 24

def read_node_addresses(filename="servers.txt"):
    with open(filename, 'r') as f:
        lines = [line.strip().split() for line in f.readlines() if line.strip() and not line.startswith('#')]
    return lines

def verify_and_process_files(file1, file2):
    if os.path.basename(file1) != os.path.basename(file2):
        raise ValueError("File names do not match")

    print(f"Accumulating {file1} += {file2}")

    try:
        # Construct the command
        command = f"{app_path}/build/bin/sum_correlations {file1} {file2}"

        # Run the command
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    except subprocess.CalledProcessError as e:
        # Handle the case where the command failed
        print(f"Command failed with exit code {e.returncode}")
        print("Error message:", e.stderr.decode())
        raise ValueError("Summation failed")

def process_files_in_threads(out_files, work_files):
    threads = []
    for i in range(0, len(out_files)):
        if i < len(work_files):  # Ensuring there's a corresponding file in work_files
            thread = threading.Thread(target=verify_and_process_files, args=(out_files[i], work_files[i]))
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
    servers = read_node_addresses()
    username = 'your_username'
    workspace = './workspace/'
    outputs_dir = './outputs/'

    if os.path.exists(outputs_dir):
        print(f"Before starting a new collection, you must manually delete the {outputs_dir} folder.  This is required to avoid losing data by accident.")
        return

    os.makedirs(outputs_dir, exist_ok=True)

    for hostname, _ in servers:
        print("Recreating empty workspace folder...")
        if os.path.exists(workspace):
            shutil.rmtree(workspace)
        os.makedirs(workspace, exist_ok=True)

        print(f"Fetching files from {hostname}...")
        os.system(f"rsync -av -e ssh {hostname}:{app_path}/correlations_block_*.zstd {workspace}")

        print(f"Checking files from {hostname}...")
        work_files = [os.path.join(workspace, f) for f in os.listdir(workspace) if f.startswith('correlations_block_')]
        work_files.sort(key=natural_sort_key)

        out_files = [os.path.join(outputs_dir, f) for f in os.listdir(outputs_dir) if f.startswith('correlations_block_')]
        out_files.sort(key=natural_sort_key)

        if len(out_files) == 0:
            print(f"This is the first set of files. Moving them to outputs folder...")
            for source_file in work_files:
                shutil.move(source_file, os.path.join(outputs_dir, os.path.basename(source_file)))
            continue

        if len(work_files) != len(out_files):
            raise ValueError("Some files are missing")

        process_files_in_threads(out_files, work_files)

if __name__ == "__main__":
    main()
