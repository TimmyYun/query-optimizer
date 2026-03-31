# ============================================================
# DSB Benchmark: Setup, Data Generation & Query Generation
# ============================================================

# --- 0. INITIAL SETUP ---
WORKLOAD_COUNT=40
UPDATE_STREAMS=5



# --- 1. Clone DSB repository ---
git clone https://github.com/microsoft/dsb.git

cd ./dsb/code/tools

# --- 2. Create custom query templates ---
# These templates define parameterized WHERE clauses
# for selectivity estimation experiments on 3 target columns:
#   - catalog_returns.cr_item_sk
#   - catalog_returns.cr_returned_time_sk
#   - web_sales.ws_item_sk

mkdir ./query_templates

cat > ./query_templates/custom_cr_item_sk.tpl << 'EOF'
define ITEM_START = random(1, 17999, uniform);
define RANGE_ = random(700, 3000, uniform);

WHERE cr_item_sk BETWEEN [ITEM_START] AND [ITEM_START] + [RANGE_];
EOF

cat > ./query_templates/custom_cr_returned_time_sk.tpl << 'EOF'
define RETURNED_TIME_START = random(2, 86397, uniform);
define RANGE_ = random(1000, 4000, uniform);

WHERE cr_returned_time_sk BETWEEN [RETURNED_TIME_START] AND [RETURNED_TIME_START] + [RANGE_];
EOF

cat > ./query_templates/custom_ws_item_sk.tpl << 'EOF'
define ITEM_START = random(1, 17999, uniform);
define RANGE_ = random(400, 2000, uniform);

WHERE ws_item_sk BETWEEN [ITEM_START] AND [ITEM_START] + [RANGE_];
EOF

cat > ./query_templates/postgres.tpl << 'EOF' 
--
-- Legal Notice
--
-- This document and associated source code (the "Work") is a part of a
-- benchmark specification maintained by the TPC.
--
-- The TPC reserves all right, title, and interest to the Work as provided
-- under U.S. and international laws, including without limitation all patent
-- and trademark rights therein.
--
-- No Warranty
--
-- 1.1 TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THE INFORMATION
--     CONTAINED HEREIN IS PROVIDED "AS IS" AND WITH ALL FAULTS, AND THE
--     AUTHORS AND DEVELOPERS OF THE WORK HEREBY DISCLAIM ALL OTHER
--     WARRANTIES AND CONDITIONS, EITHER EXPRESS, IMPLIED OR STATUTORY,
--     INCLUDING, BUT NOT LIMITED TO, ANY (IF ANY) IMPLIED WARRANTIES,
--     DUTIES OR CONDITIONS OF MERCHANTABILITY, OF FITNESS FOR A PARTICULAR
--     PURPOSE, OF ACCURACY OR COMPLETENESS OF RESPONSES, OF RESULTS, OF
--     WORKMANLIKE EFFORT, OF LACK OF VIRUSES, AND OF LACK OF NEGLIGENCE.
--     ALSO, THERE IS NO WARRANTY OR CONDITION OF TITLE, QUIET ENJOYMENT,
--     QUIET POSSESSION, CORRESPONDENCE TO DESCRIPTION OR NON-INFRINGEMENT
--     WITH REGARD TO THE WORK.
-- 1.2 IN NO EVENT WILL ANY AUTHOR OR DEVELOPER OF THE WORK BE LIABLE TO
--     ANY OTHER PARTY FOR ANY DAMAGES, INCLUDING BUT NOT LIMITED TO THE
--     COST OF PROCURING SUBSTITUTE GOODS OR SERVICES, LOST PROFITS, LOSS
--     OF USE, LOSS OF DATA, OR ANY INCIDENTAL, CONSEQUENTIAL, DIRECT,
--     INDIRECT, OR SPECIAL DAMAGES WHETHER UNDER CONTRACT, TORT, WARRANTY,
--     OR OTHERWISE, ARISING IN ANY WAY OUT OF THIS OR ANY OTHER AGREEMENT
--     RELATING TO THE WORK, WHETHER OR NOT SUCH AUTHOR OR DEVELOPER HAD
--     ADVANCE NOTICE OF THE POSSIBILITY OF SUCH DAMAGES.
--
-- Contributors:
--
define __LIMITA = "";
define __LIMITB = "";
define __LIMITC = "limit %d";
define _END = "";
EOF

# --- 3. Build dsdgen and dsqgen ---
make

# --- 4. Generate base data (1GB scale factor) ---
# Creates ~23 .dat files in ./data1gb
mkdir ./data1gb
./dsdgen -scale 1 -dir ./data1gb

# --- 5. Generate 1000 refresh streams ---
# Each stream produces ~23 staging files (s_*.dat, delete_*.dat)
# Used to simulate data drift over time

mkdir ./data1gbupdates
for i in $(seq 1 "$UPDATE_STREAMS"); do
  ./dsdgen -scale 1 -dir ./data1gbupdates -update "$i"
done

# --- 6. Generate 400K queries per template ---
# dsqgen uses -stream N to produce N queries with different random seeds
# sed removes dsqgen metadata lines (name, param, blank lines)

./dsqgen \
  -template custom_ws_item_sk.tpl \
  -directory ./query_templates \
  -dialect postgres \
  -scale 1 \
  -stream "$WORKLOAD_COUNT" \
  -FILTER | sed '/^name /d; /^param /d; /^[[:space:]]*$/d' \
  > custom_ws_item_sk_initial.sql

./dsqgen \
  -template custom_cr_item_sk.tpl \
  -directory ./query_templates \
  -dialect postgres \
  -scale 1 \
  -stream "$WORKLOAD_COUNT" \
  -FILTER | sed '/^name /d; /^param /d; /^[[:space:]]*$/d' \
  > custom_cr_item_sk_initial.sql

./dsqgen \
  -template custom_cr_returned_time_sk.tpl \
  -directory ./query_templates \
  -dialect postgres \
  -scale 1 \
  -stream "$WORKLOAD_COUNT" \
  -FILTER | sed '/^name /d; /^param /d; /^[[:space:]]*$/d' \
  > custom_cr_returned_time_sk_initial.sql

# --- 7. Remove duplicate queries ---
# dsqgen can produce identical queries for different streams
# This script keeps only unique queries

mkdir ./deduped_queries

cat > duplicate_deleter.py << 'PYEOF'
initial_files = [
    'custom_cr_item_sk_initial',
    'custom_cr_returned_time_sk_initial',
    'custom_ws_item_sk_initial'
]

def duplicate_deleter(input_file: str, output_file: str):
    with open(input_file) as f:
        lines = [line.strip() for line in f if line.strip()]

    seen = set()
    unique = []
    duplicates = 0

    for line in lines:
        if line in seen:
            duplicates += 1
        else:
            seen.add(line)
            unique.append(line)

    with open(output_file, "w") as f:
        for line in unique:
            f.write(line + "\n")

    print(f"Before: {len(lines)}")
    print(f"Duplicates removed: {duplicates}")
    print(f"After: {len(unique)}")

for file in initial_files:
    input_path = f"./{file}.sql"
    output_path = f"./deduped_queries/{file}_deduped.sql"
    print(f"Processing {input_path}...")
    duplicate_deleter(input_path, output_path)
    print(f"Saved to {output_path}\n")
PYEOF

python3 duplicate_deleter.py
