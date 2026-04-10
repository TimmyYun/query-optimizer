import random
import csv

TEMPLATES = {
    "PostHistoryLengthText": {
        "start_range": (0, 31278),
        "size_range": (100, 300),
        "max_value": 31278,
    }
}

MASTER_SEED = 42
N = 400_000

for name, template in TEMPLATES.items():
    rng = random.Random(MASTER_SEED)
    seen = set()
    rows = []

    for _ in range(N):
        low = rng.randint(*template["start_range"])
        size = rng.randint(*template["size_range"])
        high = min(low + size, template["max_value"])

        pair = (low, high)
        if pair not in seen:
            seen.add(pair)
            rows.append(pair)

    path = f"data/STATS-CEB/{name}.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow(["low", "high"])
        writer.writerows(rows)

    print(f"{name}: {len(rows)} unique pairs → {path}")