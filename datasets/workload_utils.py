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
