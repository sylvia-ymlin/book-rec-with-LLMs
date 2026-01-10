import pexpect
import os
import sys

def sync_to_server():
    host = "connect.nmb2.seetacloud.com"
    port = "26578"
    user = "root"
    password = "9Dml+WZeqp5b"
    
    local_path = os.getcwd()
    remote_path = "/root/book-rec-with-LLMs"
    
    print(f"Syncing to {user}@{host}:{port}...")
    
    # 1. Create remote directory
    # ssh -p PORT user@host "mkdir -p remote_path"
    cmd_mkdir = f"ssh -p {port} {user}@{host} 'mkdir -p {remote_path}'"
    child = pexpect.spawn(cmd_mkdir)
    i = child.expect(['password:', 'continue connecting', pexpect.EOF, pexpect.TIMEOUT])
    
    if i == 1: # verify host key
        child.sendline('yes')
        child.expect('password:')
        
    if i == 0 or i == 1:
        child.sendline(password)
        child.expect(pexpect.EOF)
    
    print("Remote directory created.")
    
    # 2. Sync Code (src, scripts, requirements.txt)
    # Using scp -r -P PORT ...
    # Exclude files handled by .gitignore if using rsync, but scp is simpler for now
    
    # Let's sync folders individually to be safe
    # 2. Sync Code
    folders = ['src', 'scripts']
    for folder in folders:
        print(f"Syncing {folder}...")
        cmd_scp = f"scp -r -P {port} {local_path}/{folder} {user}@{host}:{remote_path}/"
        
        child = pexpect.spawn(cmd_scp, timeout=300)
        i = child.expect(['password:', 'continue connecting', pexpect.EOF, pexpect.TIMEOUT])
        if i == 1:
            child.sendline('yes')
            child.expect('password:')
        if i == 0 or i == 1:
            child.sendline(password)
            child.expect(pexpect.EOF)
        print(f"Synced {folder}.")

    # 3. Sync Data (Only necessary files)
    print("Syncing data files (rec/ and books_processed.csv)...")
    
    # Ensure data dir exists
    cmd_mkdir_data = f"ssh -p {port} {user}@{host} 'mkdir -p {remote_path}/data'"
    child = pexpect.spawn(cmd_mkdir_data)
    child.expect('password:')
    child.sendline(password)
    child.expect(pexpect.EOF)

    # Sync books_processed.csv
    print("Syncing books_processed.csv (350MB+)...")
    cmd_scp_books = f"scp -P {port} {local_path}/data/books_processed.csv {user}@{host}:{remote_path}/data/"
    child = pexpect.spawn(cmd_scp_books, timeout=1200) # 20 mins
    child.expect('password:')
    child.sendline(password)
    child.expect(pexpect.EOF)
    
    # Sync rec/ folder
    print("Syncing data/rec/ folder (1.5GB+)...")
    cmd_scp_rec = f"scp -r -P {port} {local_path}/data/rec {user}@{host}:{remote_path}/data/"
    child = pexpect.spawn(cmd_scp_rec, timeout=3600) # 60 mins
    child.expect('password:')
    child.sendline(password)
    child.expect(pexpect.EOF)
        
    print("Sync Completed!")
        
    print("Sync Completed!")

if __name__ == "__main__":
    sync_to_server()
