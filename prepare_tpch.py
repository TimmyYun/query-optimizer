import zipfile
import csv
import io
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Extract column from TPC-H tbl in zip")
    parser.add_argument("--zip", default="Data10Gb.zip", help="Path to zip file")
    parser.add_argument("--table", required=True, help="Table filename inside zip (e.g. lineitem.tbl)")
    parser.add_argument("--col", type=int, required=True, help="Column index (0-based)")
    parser.add_argument("--limit", type=int, default=0, help="Max rows to extract (0 for all)")
    parser.add_argument("--scale", type=float, default=100.0, help="Scale factor for float->int conversion")
    parser.add_argument("--out", required=True, help="Output CSV")
    args = parser.parse_args()

    zip_path = Path(args.zip)
    if not zip_path.exists():
        print(f"Error: {zip_path} not found.")
        return

    print(f"Reading {args.table} from {zip_path}...")
    
    count = 0
    with zipfile.ZipFile(zip_path, 'r') as zf:
        if args.table not in zf.namelist():
             print(f"Error: {args.table} not found in zip.")
             return
             
        with zf.open(args.table) as fin, open(args.out, 'w', newline='') as fout:
            writer = csv.writer(fout)
            text_input = io.TextIOWrapper(fin, encoding='utf-8', errors='replace')
            
            for line in text_input:
                parts = line.split('|')
                if len(parts) > args.col:
                    val_str = parts[args.col]
                    try:
                        # Convert to float then int
                        val = int(float(val_str) * args.scale)
                        writer.writerow([val])
                        count += 1
                        if args.limit > 0 and count >= args.limit:
                            break
                    except ValueError:
                        pass
                
                if count % 1000000 == 0:
                    print(f"Processed {count} rows...", end='\r')

    print(f"\nDone. Extracted {count} rows to {args.out}")

if __name__ == "__main__":
    main()
