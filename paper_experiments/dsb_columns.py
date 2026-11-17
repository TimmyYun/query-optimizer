"""
DSB expansion, part 1: profile candidate numeric columns of DSB SF=1 and select the benchmark set.

Data: "DSB benchmark/dsb/code/tools/data1gb/<table>.dat" (dsdgen, SF=1, official toolkit pinned in
"DSB benchmark/setup_dsb.sh"), '|'-separated, no header; column names from the official schema
(the same `schema_headers` as "DSB benchmark/DSB_GENERATOR.ipynb").
Decimals (DECIMAL(7,2)) are stored as integer cents (x100, lossless), as prepare_benchmark.py does.
NULLs are dropped (no range predicate selects a NULL).

    poetry run python -m paper_experiments.dsb_columns            # profile all candidates
    poetry run python -m paper_experiments.dsb_columns --manifest # + DATASET_MANIFEST.md for SELECTED

Outputs (results/paper_experiments/dsb_expanded/):
    dataset_profile_candidates.csv   every candidate column, every statistic
    dataset_manifest.csv             the selected columns (+ actual meta.pkl values once built)
    DATASET_MANIFEST.md              justification per selected column, rejected candidates + reasons
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
DSB_DATA = ROOT / "DSB benchmark" / "dsb" / "code" / "tools" / "data1gb"
TEMPLATES = ROOT / "DSB benchmark" / "dsb" / "query_templates_pg"
OUT = ROOT / "results" / "paper_experiments" / "dsb_expanded"
DOMAIN_LIMIT = 5_000_000
SAMPLE_SIZE = 100_000
FD_CAP = 2000

SCHEMA = {
    "catalog_returns": ['cr_returned_date_sk', 'cr_returned_time_sk', 'cr_item_sk', 'cr_refunded_customer_sk', 'cr_refunded_cdemo_sk', 'cr_refunded_hdemo_sk', 'cr_refunded_addr_sk', 'cr_returning_customer_sk', 'cr_returning_cdemo_sk', 'cr_returning_hdemo_sk', 'cr_returning_addr_sk', 'cr_call_center_sk', 'cr_catalog_page_sk', 'cr_ship_mode_sk', 'cr_warehouse_sk', 'cr_reason_sk', 'cr_order_number', 'cr_return_quantity', 'cr_return_amount', 'cr_return_tax', 'cr_return_amt_inc_tax', 'cr_fee', 'cr_return_ship_cost', 'cr_refunded_cash', 'cr_reversed_charge', 'cr_store_credit', 'cr_net_loss'],
    "web_sales": ['ws_sold_date_sk', 'ws_sold_time_sk', 'ws_ship_date_sk', 'ws_item_sk', 'ws_bill_customer_sk', 'ws_bill_cdemo_sk', 'ws_bill_hdemo_sk', 'ws_bill_addr_sk', 'ws_ship_customer_sk', 'ws_ship_cdemo_sk', 'ws_ship_hdemo_sk', 'ws_ship_addr_sk', 'ws_web_page_sk', 'ws_web_site_sk', 'ws_ship_mode_sk', 'ws_warehouse_sk', 'ws_promo_sk', 'ws_order_number', 'ws_quantity', 'ws_wholesale_cost', 'ws_list_price', 'ws_sales_price', 'ws_ext_discount_amt', 'ws_ext_sales_price', 'ws_ext_wholesale_cost', 'ws_ext_list_price', 'ws_ext_tax', 'ws_coupon_amt', 'ws_ext_ship_cost', 'ws_net_paid', 'ws_net_paid_inc_tax', 'ws_net_paid_inc_ship', 'ws_net_paid_inc_ship_tax', 'ws_net_profit'],
    "store_sales": ['ss_sold_date_sk', 'ss_sold_time_sk', 'ss_item_sk', 'ss_customer_sk', 'ss_cdemo_sk', 'ss_hdemo_sk', 'ss_addr_sk', 'ss_store_sk', 'ss_promo_sk', 'ss_ticket_number', 'ss_quantity', 'ss_wholesale_cost', 'ss_list_price', 'ss_sales_price', 'ss_ext_discount_amt', 'ss_ext_sales_price', 'ss_ext_wholesale_cost', 'ss_ext_list_price', 'ss_ext_tax', 'ss_coupon_amt', 'ss_net_paid', 'ss_net_paid_inc_tax', 'ss_net_profit'],
    "catalog_sales": ['cs_sold_date_sk', 'cs_sold_time_sk', 'cs_ship_date_sk', 'cs_bill_customer_sk', 'cs_bill_cdemo_sk', 'cs_bill_hdemo_sk', 'cs_bill_addr_sk', 'cs_ship_customer_sk', 'cs_ship_cdemo_sk', 'cs_ship_hdemo_sk', 'cs_ship_addr_sk', 'cs_call_center_sk', 'cs_catalog_page_sk', 'cs_ship_mode_sk', 'cs_warehouse_sk', 'cs_item_sk', 'cs_promo_sk', 'cs_order_number', 'cs_quantity', 'cs_wholesale_cost', 'cs_list_price', 'cs_sales_price', 'cs_ext_discount_amt', 'cs_ext_sales_price', 'cs_ext_wholesale_cost', 'cs_ext_list_price', 'cs_ext_tax', 'cs_coupon_amt', 'cs_ext_ship_cost', 'cs_net_paid', 'cs_net_paid_inc_tax', 'cs_net_paid_inc_ship', 'cs_net_paid_inc_ship_tax', 'cs_net_profit'],
    "store_returns": ['sr_returned_date_sk', 'sr_return_time_sk', 'sr_item_sk', 'sr_customer_sk', 'sr_cdemo_sk', 'sr_hdemo_sk', 'sr_addr_sk', 'sr_store_sk', 'sr_reason_sk', 'sr_ticket_number', 'sr_return_quantity', 'sr_return_amt', 'sr_return_tax', 'sr_return_amt_inc_tax', 'sr_fee', 'sr_return_ship_cost', 'sr_refunded_cash', 'sr_reversed_charge', 'sr_store_credit', 'sr_net_loss'],
    "web_returns": ['wr_returned_date_sk', 'wr_returned_time_sk', 'wr_item_sk', 'wr_refunded_customer_sk', 'wr_refunded_cdemo_sk', 'wr_refunded_hdemo_sk', 'wr_refunded_addr_sk', 'wr_returning_customer_sk', 'wr_returning_cdemo_sk', 'wr_returning_hdemo_sk', 'wr_returning_addr_sk', 'wr_web_page_sk', 'wr_reason_sk', 'wr_order_number', 'wr_return_quantity', 'wr_return_amt', 'wr_return_tax', 'wr_return_amt_inc_tax', 'wr_fee', 'wr_return_ship_cost', 'wr_refunded_cash', 'wr_reversed_charge', 'wr_account_credit', 'wr_net_loss'],
    "inventory": ['inv_date_sk', 'inv_item_sk', 'inv_warehouse_sk', 'inv_quantity_on_hand'],
    "customer": ['c_customer_sk', 'c_customer_id', 'c_current_cdemo_sk', 'c_current_hdemo_sk', 'c_current_addr_sk', 'c_first_shipto_date_sk', 'c_first_sales_date_sk', 'c_salutation', 'c_first_name', 'c_last_name', 'c_preferred_cust_flag', 'c_birth_day', 'c_birth_month', 'c_birth_year', 'c_birth_country', 'c_login', 'c_email_address', 'c_last_review_date'],
    "item": ['i_item_sk', 'i_item_id', 'i_rec_start_date', 'i_rec_end_date', 'i_item_desc', 'i_current_price', 'i_wholesale_cost', 'i_brand_id', 'i_brand', 'i_class_id', 'i_class', 'i_category_id', 'i_category', 'i_manufact_id', 'i_manufact', 'i_size', 'i_formulation', 'i_color', 'i_units', 'i_container', 'i_manager_id', 'i_product_name'],
    "customer_demographics": ['cd_demo_sk', 'cd_gender', 'cd_marital_status', 'cd_education_status', 'cd_purchase_estimate', 'cd_credit_rating', 'cd_dep_count', 'cd_dep_employed_count', 'cd_dep_college_count'],
}

# column -> scale to integer (1 = integer column, 100 = DECIMAL(7,2) in cents)
CANDIDATES = {
    "catalog_returns": {"cr_item_sk": 1, "cr_returned_time_sk": 1, "cr_returned_date_sk": 1, "cr_return_quantity": 1,
                        "cr_return_amount": 100, "cr_order_number": 1},
    "web_sales": {"ws_item_sk": 1, "ws_sold_time_sk": 1, "ws_quantity": 1},
    "store_sales": {"ss_item_sk": 1, "ss_sold_time_sk": 1, "ss_sold_date_sk": 1, "ss_customer_sk": 1, "ss_quantity": 1,
                    "ss_list_price": 100, "ss_sales_price": 100, "ss_wholesale_cost": 100, "ss_net_profit": 100,
                    "ss_ext_sales_price": 100, "ss_ticket_number": 1},
    "catalog_sales": {"cs_item_sk": 1, "cs_sold_date_sk": 1, "cs_ship_date_sk": 1, "cs_quantity": 1,
                      "cs_wholesale_cost": 100, "cs_list_price": 100, "cs_ext_discount_amt": 100},
    "store_returns": {"sr_item_sk": 1, "sr_return_time_sk": 1, "sr_customer_sk": 1, "sr_return_quantity": 1,
                      "sr_return_amt": 100},
    "web_returns": {"wr_item_sk": 1, "wr_returned_time_sk": 1, "wr_return_quantity": 1, "wr_return_amt": 100},
    "inventory": {"inv_quantity_on_hand": 1, "inv_item_sk": 1, "inv_date_sk": 1},
    "customer": {"c_birth_year": 1, "c_first_sales_date_sk": 1},
    "item": {"i_current_price": 100, "i_manager_id": 1, "i_manufact_id": 1, "i_brand_id": 1},
    "customer_demographics": {"cd_purchase_estimate": 1, "cd_dep_count": 1},
}

# The benchmark set: 3 thesis columns + 5 from other tables (justifications in DATASET_MANIFEST.md)
SELECTED = {
    "catalog_returns.cr_item_sk": "thesis column; foreign key of returned items, skewed item popularity (thesis §4.2.2)",
    "catalog_returns.cr_returned_time_sk": "thesis column (written cr_ret_time_sk in the thesis; the official name is "
                                           "cr_returned_time_sk); time-of-day key, flat/low structure baseline",
    "web_sales.ws_item_sk": "thesis column; item key of web sales, near-uniform key distribution",
}
# 5 columns from OTHER tables, chosen from dataset_profile_candidates.csv for diversity
SELECTED_EXTRA = {
    "store_sales.ss_net_profit": "measure with negative values, largest NDV (444k) and FD k (~1429, closest to the "
                                 "2000 cap of all candidates), 2.75M rows (sample 3.6 %); official BETWEEN (query013)",
    "store_sales.ss_sales_price": "skewed price measure with a heavy hitter (top-1 value = 11 % of rows) on a sparse "
                                  "lattice (NDV 5.5k over a 30k-cent domain); official BETWEEN (query013)",
    "inventory.inv_quantity_on_hand": "largest table (11.2M rows, sample 0.9 %), dense small domain (1001 values), "
                                      "near-uniform; official BETWEEN with parameters (query039)",
    "catalog_sales.cs_sold_date_sk": "date key (2148 days, dense), 1.43M rows, seasonal/periodic structure; dates are "
                                     "the classic range predicate",
    "store_returns.sr_return_amt": "most skewed measure (skew 3.4, kurtosis high) with a 10.9 % point mass, very sparse "
                                   "(NDV 138k over a 2.79M-cent domain), FD k ~950",
}
SELECTED_ALL = {**SELECTED, **SELECTED_EXTRA}

REJECTED = {
    "customer.c_birth_year": "N = 96.5k < 100k sample: the sample is the whole column; domain 69 < 100 fixed buckets "
                             "(width-1 buckets are exact) — degenerate for sample-based estimation",
    "item.*": "item has 18k rows: sample = whole table (degenerate); i_brand_id domain 9.0M exceeds the dense-array limit",
    "customer_demographics.*": "cross-product dimension: every value has exactly equal frequency (cd_purchase_estimate "
                               "20 values x 5 % each, cd_dep_count 7 values) — trivial for every histogram",
    "*_quantity (ws/cs/ss/sr/cr/wr)": "100-value domains, either uniform (sales) or one shape (returns); "
                                      "inv_quantity_on_hand covers the small-dense-domain case with far more rows",
    "*_wholesale_cost / *_list_price": "20 / 397 distinct lattice values; already covered by the existing DSB columns "
                                       "web_sales.ws_wholesale_cost / ws_list_price in data/dsb",
    "other *_item_sk (ss/cs/sr/wr/inv)": "same 18k item domain and popularity skew as cr_item_sk / ws_item_sk "
                                        "(thesis columns); inv_item_sk is exactly uniform",
    "*_time_sk (ws/ss/sr/wr)": "same time-of-day key as cr_returned_time_sk; ss/sr are truncated to opening hours",
    "ss_ext_sales_price / cs_ext_discount_amt / *_return_amount": "skewed amount measures similar to sr_return_amt "
                                                                 "(kept the most skewed one)",
    "*_order_number / ss_ticket_number / *_customer_sk": "near-uniform surrogate keys; uniform keys are already covered "
                                                         "by ws_item_sk / inv_quantity_on_hand",
    "*_date_sk except cs_sold_date_sk": "same date structure; one date key is enough",
}


def load_column(table: str, column: str, scale: int) -> tuple[np.ndarray, int]:
    """Non-null values of table.column as int64 (decimals x scale), and the table's row count."""
    idx = SCHEMA[table].index(column)
    s = pd.read_csv(DSB_DATA / f"{table}.dat", sep="|", header=None, usecols=[idx], dtype="float64",
                    engine="c").iloc[:, 0]
    n_rows = len(s)
    s = s.dropna().to_numpy(dtype=np.float64)
    return np.rint(s * scale).astype(np.int64), n_rows


def fd_bins_full(values: np.ndarray, n_rows: int) -> int:
    """Repo FD formula (data/datasets.freedman_diaconis_bins) on full-column quartiles (profile only;
    the k actually used is computed from the 100k sample in meta.pkl)."""
    q25, q75 = np.quantile(values, [0.25, 0.75])
    iqr = q75 - q25
    if iqr <= 0:
        return 10
    h = 2 * iqr / (n_rows ** (1 / 3))
    return max(1, min(int((values.max() - values.min()) / h), FD_CAP))


def between_columns() -> dict:
    """column -> number of official pg templates that filter it with '<col> BETWEEN'."""
    out = {}
    for t in TEMPLATES.rglob("*.tpl"):
        if t.name == "postgres.tpl":
            continue
        for c in set(m.lower() for m in re.findall(r"\b([a-z]{1,4}_[a-z_]+)\s+between", t.read_text(errors="ignore"), re.I)):
            out[c] = out.get(c, 0) + 1
    return out


def profile(values: np.ndarray, n_rows: int) -> dict:
    mn, mx = int(values.min()), int(values.max())
    uniq, counts = np.unique(values, return_counts=True)
    counts_sorted = np.sort(counts)[::-1]
    N = len(values)
    top1pct = max(1, int(np.ceil(0.01 * len(uniq))))
    return {
        "n_rows_table": n_rows, "N": N, "null_frac": round(1 - N / n_rows, 6),
        "ndv": len(uniq), "mn": mn, "mx": mx, "domain": mx - mn + 1,
        "density_ndv_per_domain": round(len(uniq) / (mx - mn + 1), 6),
        "skew": float(stats.skew(values)), "kurtosis": float(stats.kurtosis(values)),
        "top1_mass": float(counts_sorted[0] / N), "top10_mass": float(counts_sorted[:10].sum() / N),
        "top1pct_values_mass": float(counts_sorted[:top1pct].sum() / N),
        "fd_k_profile": fd_bins_full(values, N),
        "sample_fraction": round(min(1.0, SAMPLE_SIZE / N), 4),
    }


def run_profile():
    OUT.mkdir(parents=True, exist_ok=True)
    betw = between_columns()
    rows = []
    for table, cols in CANDIDATES.items():
        for col, scale in cols.items():
            v, n_rows = load_column(table, col, scale)
            p = profile(v, n_rows)
            p.update({"key": f"{table}.{col}", "table": table, "column": col, "scale": scale,
                      "official_between_templates": betw.get(col, 0),
                      "domain_ok": p["domain"] <= DOMAIN_LIMIT})
            rows.append(p)
            print(f"{table}.{col}: N={p['N']:,} ndv={p['ndv']:,} dom={p['domain']:,} skew={p['skew']:.2f} "
                  f"top1={p['top1_mass']:.4f} fd={p['fd_k_profile']} between={p['official_between_templates']}",
                  flush=True)
    df = pd.DataFrame(rows)
    first = ["key", "table", "column", "scale"]
    df = df[first + [c for c in df.columns if c not in first]]
    df.to_csv(OUT / "dataset_profile_candidates.csv", index=False)
    return df


def write_manifest_files(profile_df: pd.DataFrame):
    """dataset_manifest.csv + DATASET_MANIFEST.md for SELECTED_ALL (actual meta.pkl values added when built)."""
    import pickle
    rows = []
    for key, why in SELECTED_ALL.items():
        r = profile_df[profile_df["key"] == key].iloc[0].to_dict()
        r["role"] = "thesis" if key in SELECTED else "extra"
        r["justification"] = why
        meta = ROOT / "data" / "dsb_expanded" / key / "meta.pkl"
        if meta.exists():
            mn, mx, N, freq, sample, k, skew, kurt = pickle.load(open(meta, "rb"))
            from paper_experiments.provenance import file_sha256
            r.update({"meta_N": int(N), "meta_mn": int(mn), "meta_mx": int(mx), "meta_fd_k": int(k),
                      "meta_sample": len(sample), "meta_sha256": file_sha256(meta.relative_to(ROOT))})
        rows.append(r)
    m = pd.DataFrame(rows)
    m.to_csv(OUT / "dataset_manifest.csv", index=False)

    def f(x, nd=3):
        return f"{x:,.{nd}f}" if isinstance(x, float) else f"{x:,}" if isinstance(x, (int, np.integer)) else str(x)

    lines = [
        "# DSB expanded — dataset manifest",
        "",
        "Source: DSB (Ding et al., VLDB 2021) toolkit pinned in `DSB benchmark/setup_dsb.sh`, `dsdgen -scale 1`",
        "(SF = 1, the same data as the existing `data/dsb` columns), files `DSB benchmark/dsb/code/tools/data1gb/*.dat`.",
        "Decimals are stored as integer cents (x100, lossless); NULLs dropped. Profile statistics are over the full",
        "non-null column; `fd_k_profile` uses full-column quartiles, the k actually used (`meta_fd_k`) is computed by",
        "the repo pipeline from the 100k sample (`data/datasets.freedman_diaconis_bins`, n = N, cap 2000).",
        "Profiled candidates (47): `dataset_profile_candidates.csv`. Generator: `paper_experiments/dsb_columns.py`.",
        "",
        "Column name check: the thesis writes `cr_ret_time_sk` (§4.2.2) and `cr_returned_time_sk` (Tables 4.20, Fig. 4-12);",
        "the official DSB / TPC-DS schema name is **`catalog_returns.cr_returned_time_sk`** (used here).",
        "",
        "No DSB SF=1 column reaches the FD cap of 2000 buckets (largest: ss_net_profit). The cap is therefore",
        "inactive on DSB at SF = 1 — a fact, not a selection artefact.",
        "",
        "## Selected columns",
        "",
        "| column | role | N | null % | NDV | domain | NDV/domain | skew | kurt | top-1 | top-10 | FD k (meta) | sample frac | official BETWEEN templates |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in m.iterrows():
        lines.append(
            f"| `{r['key']}` | {r['role']} | {f(int(r['N']))} | {100 * r['null_frac']:.2f} | {f(int(r['ndv']))} | "
            f"{f(int(r['domain']))} | {r['density_ndv_per_domain']:.4f} | {r['skew']:.2f} | {r['kurtosis']:.2f} | "
            f"{r['top1_mass']:.4f} | {r['top10_mass']:.4f} | {r.get('meta_fd_k', r['fd_k_profile'])} | "
            f"{r['sample_fraction']:.4f} | {int(r['official_between_templates'])} |")
    lines += ["", "### Justification", ""]
    for _, r in m.iterrows():
        lines.append(f"- **`{r['key']}`** — {r['justification']}.")
    lines += ["", "## Rejected candidates", ""]
    for k, why in REJECTED.items():
        lines.append(f"- `{k}` — {why}.")
    (OUT / "DATASET_MANIFEST.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT / 'DATASET_MANIFEST.md'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", action="store_true", help="Only rebuild the manifest from the saved profile")
    args = ap.parse_args()
    df = pd.read_csv(OUT / "dataset_profile_candidates.csv") if args.manifest else run_profile()
    write_manifest_files(df)
    print(json.dumps(SELECTED_ALL, indent=1))


if __name__ == "__main__":
    main()
