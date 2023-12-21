import subprocess

app_path = "/home/catid/sources/llama.cpp"

def read_servers(filename):
    with open(filename, 'r') as file:
        # Extract only server names, ignoring numbers
        servers = [line.split()[0] for line in file if line.strip()]
    return servers

def remove_bin_files(servers):
    # Join server names separated by comma for pdsh
    server_list = ','.join(servers)

    print(f"Servers: {server_list}")

    # Construct pdsh command
    command = f"pdsh -b -R ssh -w {server_list} 'rm -f {app_path}/correlations_*.bin'"

    # Execute the command
    try:
        subprocess.run(command, shell=True, check=True)
        print("Command executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")

servers = read_servers('servers.txt')
remove_bin_files(servers)
