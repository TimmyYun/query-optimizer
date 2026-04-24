import re

with open('benchmark_utils.py', 'r') as f:
    content = f.read()

# We can dynamically inject the dset path into args so that run_logic uses the specific distribution folder
# instead of the base dataset folder.
replacement = """
        # Load Data
        try:
            metadata = load_data_and_metadata(dset)
            
            # Temporarily set args.dataset to the specific distribution so workload paths resolve correctly
            original_dataset = args.dataset
            args.dataset = str(dset)
            
            approach_fn(args, out_dir, metadata)
            
            # Restore
            args.dataset = original_dataset
        except Exception as e:"""

content = re.sub(r'# Load Data\n\s*try:\n\s*metadata = load_data_and_metadata\(dset\)\n\s*approach_fn\(args, out_dir, metadata\)\n\s*except Exception as e:', replacement, content)

with open('benchmark_utils.py', 'w') as f:
    f.write(content)
