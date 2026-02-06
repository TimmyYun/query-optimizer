from dataclasses import dataclass
from typing import List, Tuple, Optional, Any

@dataclass
class Bucket:
    lo: int
    hi: int
    count: int = 0
    count: int = 0

@dataclass
class RangeQuery:
    low: int
    high: int

@dataclass
class CDFTrainRow:
    x_norm: float
    y_cdf: float
