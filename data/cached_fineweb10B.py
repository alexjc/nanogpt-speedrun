import os
import sys
from huggingface_hub import hf_hub_download
# Download the TokenMonster tokens of Fineweb10B from huggingface. This
# saves about an hour of startup time compared to regenerating them.
def get(fname):
    local_dir = os.path.join(os.path.dirname(__file__), 'fineweb-tokmon-10B')
    if not os.path.exists(os.path.join(local_dir, fname)):
        hf_hub_download(repo_id="alexjc/fineweb-tokmon-10B", filename=fname,
                        repo_type="dataset", local_dir=local_dir)
get("english-28416-balanced-v1/fineweb-tokmon_val_%06d.bin" % 0)

num_chunks = 101

if len(sys.argv) >= 2: # we can pass an argument to download less
    num_chunks = int(sys.argv[1])


from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=8) as executor:
    futures = [
        executor.submit(get, f"english-28416-balanced-v1/fineweb-tokmon_train_{i:06d}.bin")
        for i in range(1, num_chunks+1)
    ]
    for future in futures:
        future.result() # Raise any exceptions that occurred
