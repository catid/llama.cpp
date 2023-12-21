import os
import threading
import struct
import shutil

app_path = "/home/catid/sources/llama.cpp"
max_threads=24

def read_node_addresses(filename="servers.txt"):
    with open(filename, 'r') as f:
        lines = [line.strip().split() for line in f.readlines() if line.strip() and not line.startswith('#')]
    return lines

def verify_and_process_files(file1, file2):
    if os.path.basename(file1) != os.path.basename(file2):
        raise ValueError("File names do not match")

    # Check file sizes and first 4 bytes
    if not verify_files(file1, file2):
        raise ValueError("Files do not meet verification criteria")

    print(f"Accumulating {file1} += {file2}")

    # Process files in blocks
    process_in_blocks(file1, file2)

def verify_files(file1, file2):
    with open(file1, 'rb') as f1, open(file2, 'rb') as f2:
        first4_f1 = f1.read(4)
        first4_f2 = f2.read(4)

        f1.seek(0, 2)  # Move to end of file
        f2.seek(0, 2)
        size_f1 = f1.tell()
        size_f2 = f2.tell()

        return (size_f1 == size_f2) and (first4_f1 == first4_f2)

def process_block(file1, file2, start, end):
    with open(file1, 'r+b') as f1, open(file2, 'rb') as f2:
        f1.seek(start)
        f2.seek(start)
        while start < end:
            data1 = struct.unpack('I', f1.read(4))[0]
            data2 = struct.unpack('I', f2.read(4))[0]
            f1.seek(start)
            f1.write(struct.pack('I', data1 + data2))
            start += 4

def process_in_blocks(file1, file2):
    block_size = 1024 * 4  # 1024 32-bit values
    threads = []
    start = 4  # Skip the first 4 bytes

    with open(file1, 'rb') as f:
        f.seek(0, 2)  # Move to end of file
        size = f.tell()

    while start < size:
        end = min(start + block_size, size)
        thread = threading.Thread(target=process_block, args=(file1, file2, start, end))
        threads.append(thread)
        thread.start()

        start += block_size

        # Limit to 16 threads at a time
        if len(threads) >= max_threads:
            for t in threads:
                t.join()
            threads = []

    # Wait for any remaining threads to finish
    for t in threads:
        t.join()

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def main():
    servers = read_node_addresses()
    username = 'your_username'
    workspace = './workspace/'
    outputs_dir = './outputs/'

    os.makedirs(outputs_dir, exist_ok=True)

    for hostname, _ in servers:
        print("Recreating empty workspace folder...")
        if os.path.exists(workspace):
            shutil.rmtree(workspace)
        os.makedirs(workspace, exist_ok=True)

        print(f"Fetching files from {hostname}...")
        os.system(f"rsync -avz -e ssh {hostname}:{app_path}/correlations_block_*.bin {workspace}")

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

        for i in range(0, len(out_files), 2):
            print(f"Processing block {i}...")
            verify_and_process_files(out_files[i], work_files[i])

if __name__ == "__main__":
    main()
