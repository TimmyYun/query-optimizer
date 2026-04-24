import glob
import re

def patch_file(filename):
    with open(filename, 'r') as f:
        content = f.read()

    # Find the block where workload is resolved
    # potential_path = Path(args.dataset) / f"workload_driven_{args.workload}.csv"
    # Change to look for workload_{args.workload}.csv or exact filename
    content = content.replace('f"workload_driven_{args.workload}.csv"', 'f"workload_{args.workload}.csv"')
    
    with open(filename, 'w') as f:
        f.write(content)

for script in glob.glob("run_*.py"):
    patch_file(script)
