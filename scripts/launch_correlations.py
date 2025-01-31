import subprocess
import os
import sys
import threading
import logging

app_path = "/home/catid/sources/llama.cpp"
model_path = "models/Mistral-7B-Instruct-v0.2-GGUF/mistral-7b-instruct-v0.2.Q8_0.gguf"
data_path = "data/wikitext-2-raw/wiki.test.raw"

# tools

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def read_node_addresses(filename="servers.txt"):
    with open(filename, 'r') as f:
        lines = [line.strip().split() for line in f.readlines() if line.strip() and not line.startswith('#')]
    return lines

def log_output(process, addr):
    for line in iter(process.stdout.readline, ""):
        logger.info(f"[{addr}] {line.rstrip()}")

# data

def copy_data_to_machine(host, abs_path):
    parent_path = os.path.dirname(abs_path)

    cmd = f"rsync -avz --rsync-path='mkdir -p {parent_path} && rsync' {abs_path}/ {host}:{abs_path}"
    try:
        subprocess.run(cmd, shell=True, check=True)
        print(f"Successfully copied {abs_path} to {host}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to copy {abs_path} to {host}: {e}")

def copy_data_to_machines(node_addresses):
    abs_path = os.path.join(app_path, data_path)

    threads = []
    for host, _ in node_addresses:
        thread = threading.Thread(target=copy_data_to_machine, args=(host, abs_path))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

# model

def copy_model_to_machine(host, parent_path, abs_path):
    cmd = f"rsync -avz --rsync-path='mkdir -p {parent_path} && rsync' {parent_path}/ {host}:{parent_path}"
    try:
        subprocess.run(cmd, shell=True, check=True)
        print(f"Successfully copied {abs_path} to {host}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to copy {abs_path} to {host}: {e}")

def copy_model_to_machines(node_addresses):
    abs_path = os.path.join(app_path, model_path)
    parent_path = os.path.dirname(abs_path)

    threads = []
    for host, _ in node_addresses:
        thread = threading.Thread(target=copy_model_to_machine, args=(host, parent_path, abs_path))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

# servers

def launch_servers(node_addresses, total_chunks):
    processes = []
    log_threads = []

    # Calculate total thread count
    total_threads = sum(int(thread_count) for _, thread_count in node_addresses)

    copy_data_to_machines(node_addresses)
    copy_model_to_machines(node_addresses)

    start_chunk = 0
    for host, thread_count_str in node_addresses:
        thread_count = int(thread_count_str)

        server_chunks = round(total_chunks * thread_count / total_threads)
        if start_chunk + server_chunks > total_chunks:
            server_chunks = total_chunks - start_chunk
        if server_chunks <= 0:
            break

        cmd = f"pdsh -b -R ssh -w {host} \"cd {app_path} && ./build/bin/perplexity -m {model_path} -f {data_path} --chunks {server_chunks} --chunk-start {start_chunk} -t {thread_count}\""
        logger.info(f"Running command: {cmd}")

        start_chunk += server_chunks

        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        log_thread = threading.Thread(target=log_output, args=(process, host))
        log_thread.start()

        processes.append(process)
        log_threads.append(log_thread)

    return processes, log_threads

def launch_servers_on_machines(node_addresses, total_chunks):
    try:
        logger.info("Launching remote shells...")
        processes, log_threads = launch_servers(node_addresses, total_chunks)

        logger.info("Waiting for termination...")
        for process in processes:
            process.wait()
        for log_thread in log_threads:
            log_thread.join()

        logger.info("Terminated...")
    except KeyboardInterrupt:
        logger.info("\nTerminating remote shells...")
        for process in processes:
            process.terminate()

    logger.info("Terminated.")

# git

def run_git_and_build(node_addresses, app_path, build_threads):
    processes = []
    log_threads = []

    for host, thread_count_str in node_addresses:
        thread_count = int(thread_count_str)

        cmd = f"pdsh -b -R ssh -w {host} \"cd {app_path} && git pull && mkdir -p build && cd build && cmake -DLLAMA_CUBLAS=ON .. && make -j{thread_count}\""
        logger.info(f"Running command: {cmd}")

        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        log_thread = threading.Thread(target=log_output, args=(process, host))
        log_thread.start()

        processes.append(process)
        log_threads.append(log_thread)

    return processes, log_threads

def run_git_and_build_on_machines(node_addresses):
    try:
        logger.info("Running git pull and build on all servers...")
        processes, log_threads = run_git_and_build(node_addresses, app_path, 16)

        logger.info("Waiting for all processes to finish...")
        for process in processes:
            process.wait()
        for log_thread in log_threads:
            log_thread.join()

        logger.info("All processes finished.")
    except KeyboardInterrupt:
        logger.info("\nTerminating processes...")
        for process in processes:
            process.terminate()
        for log_thread in log_threads:
            log_thread.join()

# main

def main():
    if len(sys.argv) < 2:
        logger.error("Usage: python script.py <total_chunks>")
        sys.exit(1)

    total_chunks = int(sys.argv[1])
    node_addresses = read_node_addresses()

    run_git_and_build_on_machines(node_addresses)
    launch_servers_on_machines(node_addresses, total_chunks)

if __name__ == "__main__":
    main()
