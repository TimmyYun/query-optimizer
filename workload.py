import numpy as np
import json
from pathlib import Path
from typing import List

class RangeQuery:
    def __init__(self, low: int, high: int):
        self.low = low
        self.high = high

    def to_dict(self):
        return {"low": int(self.low), "high": int(self.high)}

    @staticmethod
    def from_dict(d):
        return RangeQuery(d["low"], d["high"])

def generate_workload(n: int, mn: int, mx: int, seed: int = 42) -> List[RangeQuery]:
    rng = np.random.default_rng(seed)
    queries = []
    width = mx - mn
    for _ in range(n):
        l = rng.integers(mn, mx)
        # Random width up to 5% of domain
        w = rng.integers(1, max(10, width // 20)) 
        r = min(mx, l + w)
        queries.append(RangeQuery(l, r))
    return queries

def save_workload(queries: List[RangeQuery], output_path: Path):
    data = [q.to_dict() for q in queries]
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)

def load_workload(input_path: Path) -> List[RangeQuery]:
    with open(input_path, "r") as f:
        data = json.load(f)
    return [RangeQuery.from_dict(d) for d in data]

def prepare_workload(dataset_dir: Path, n_queries: int, force_regeneration: bool = False) -> Path:
    workload_path = dataset_dir / "workload.json"
    
    if not workload_path.exists() or force_regeneration:
        stats_path = dataset_dir / "stats.json"
        if not stats_path.exists():
            raise FileNotFoundError(f"Stats file not found at {stats_path}. Generate dataset first.")
            
        # Load stats to get domain
        with open(stats_path, "r") as f:
            stats = json.load(f)
        
        print(f"Generating workload: {n_queries} queries for domain [{stats['Min']}, {stats['Max']}]...")
        queries = generate_workload(n_queries, stats["Min"], stats["Max"])
        save_workload(queries, workload_path)
        
    return workload_path
