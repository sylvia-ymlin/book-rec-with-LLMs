import pexpect
import os
import sys

def sync_to_server():
    host = "connect.nmb2.seetacloud.com"
    port = "15054"
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
    folders = ['src', 'scripts', 'data']
    
    for folder in folders:
        print(f"Syncing {folder}...")
        # scp -P PORT -r local_folder user@host:remote_path/
        # Note: scp -r src ... works
        
        # We need to handle 'data' carefully. 
        # Only sync 'data/rec' and 'data/books_processed.csv' etc?
        # Let's sync entire 'data' but maybe exclude huge raw files if possible.
        # But user wants 'train.csv' etc which are in 'data/rec'.
        
        cmd_scp = f"scp -r -P {port} {local_path}/{folder} {user}@{host}:{remote_path}/"
        
        child = pexpect.spawn(cmd_scp, timeout=3000) # Long timeout for data
        i = child.expect(['password:', 'continue connecting', pexpect.EOF, pexpect.TIMEOUT])
        
        if i == 1:
            child.sendline('yes')
            child.expect('password:')
            
        if i == 0 or i == 1:
            child.sendline(password)
            # Expect EOF when transfer done
            child.expect(pexpect.EOF)
            
        print(f"Synced {folder}.")
        
    print("Sync Completed!")

if __name__ == "__main__":
    sync_to_server()
