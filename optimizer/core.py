from dataclasses import dataclass
from typing import List, Tuple, Optional, Any

@dataclass
class Bucket:
    lo: int
    hi: int
    count: int = 0
    ndv: int = 0
    exact_values: Optional[List[Tuple[int, int]]] = None # For sparse buckets

@dataclass
class RangeQuery:
    low: int
    high: int

@dataclass
class CDFTrainRow:
    x_norm: float
    y_cdf: float
