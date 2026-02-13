from main import aggregate_summaries
from pathlib import Path

if __name__ == "__main__":
    print("Regenerating summaries for all experiments...")
    aggregate_summaries("results")
    print("Done.")
