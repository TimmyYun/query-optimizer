#!/usr/bin/env bash
# ============================================================
# DSB Benchmark: Setup, Data Generation & Query Generation
# macOS (Apple Silicon / Apple clang) port of setup.sh
#
# setup.sh is the original Linux version and is left untouched.
# This script does the same steps, plus the source patches that
# the TPC-DS toolkit needs to compile with Apple clang.
#
# Safe to re-run: clone, build and generation steps are skipped
# if their output already exists.
# ============================================================
set -e

# --- 0. INITIAL SETUP ---
WORKLOAD_COUNT=${WORKLOAD_COUNT:-400000}
UPDATE_STREAMS=${UPDATE_STREAMS:-300}

# Always work relative to this script, not the caller's cwd
cd "$(dirname "$0")"

echo "============================================="
echo "DSB setup (macOS)"
echo "Workload count : $WORKLOAD_COUNT"
echo "Update streams : $UPDATE_STREAMS"
echo "============================================="

# --- 1. Clone DSB repository ---
if [ ! -d "./dsb" ]; then
    git clone --depth 1 https://github.com/microsoft/dsb.git
else
    echo ">>> ./dsb already present, skipping clone"
fi

cd ./dsb/code/tools

# --- 2. Create custom query templates ---
# These templates define parameterized WHERE clauses
# for selectivity estimation experiments on 3 target columns:
#   - catalog_returns.cr_item_sk
#   - catalog_returns.cr_returned_time_sk
#   - web_sales.ws_item_sk

mkdir -p ./query_templates

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

# --- 2b. macOS source patches (not needed on Linux) ---
# The toolkit targets glibc. Three things break under Apple clang:
#
#   1. <values.h> does not exist on macOS. Its only symbol used here is
#      MAXINT, which glibc defines as INT_MAX -- so we switch the LINUX
#      block to <limits.h> and define MAXINT identically. Using the same
#      value matters: genrand.c does arithmetic mod MAXINT, so any other
#      value would change every generated row.
#   2. <malloc.h> does not exist on macOS; malloc() lives in <stdlib.h>.
#   3. clang 15+ promoted implicit-function-declaration and int-conversion
#      from warnings to errors; this K&R-era C trips both.
#
# All three are idempotent, so re-running the script is harmless.

echo ">>> Applying macOS source patches..."

sed -i '' 's|^#define USE_VALUES_H|#define USE_LIMITS_H|' config.h

if ! grep -q "define MAXINT" porting.h; then
    sed -i '' 's|^#include <limits.h>|#include <limits.h>\
#ifndef MAXINT\
#define MAXINT INT_MAX\
#endif|' porting.h
fi

grep -rl "include <malloc.h>" --include="*.c" . | grep -v y.tab.c | \
    xargs sed -i '' 's|#include <malloc.h>|#include <stdlib.h>|' || true

# --- 3. Build dsdgen and dsqgen ---
CLANG_CFLAGS="-g -O2 -Wno-implicit-function-declaration -Wno-int-conversion -Wno-implicit-int -Wno-return-type -Wno-deprecated-non-prototype -Wno-format"

if [ ! -x ./dsdgen ] || [ ! -x ./dsqgen ]; then
    if ! make OS=LINUX LINUX_CFLAGS="$CLANG_CFLAGS"; then
        echo ""
        echo "ERROR: build failed. Check the compiler output above."
        echo "Known macOS issues, in case a new one appeared:"
        echo "  - missing <values.h>  -> config.h should use USE_LIMITS_H"
        echo "  - missing <malloc.h>  -> replace with <stdlib.h>"
        echo "  - implicit declarations / int conversions -> add -Wno-... to CLANG_CFLAGS"
        exit 1
    fi
else
    echo ">>> dsdgen and dsqgen already built, skipping make"
fi

# --- 4. Generate base data (1GB scale factor) ---
# Creates ~23 .dat files in ./data1gb
mkdir -p ./data1gb
if [ ! -f ./data1gb/web_sales.dat ]; then
    ./dsdgen -scale 1 -dir ./data1gb -force
else
    echo ">>> ./data1gb already populated, skipping"
fi

# --- 5. Generate refresh streams ---
# Each stream produces ~23 staging files (s_*.dat, delete_*.dat)
# Used to simulate data drift over time.
#
# Note: the notebook stops as soon as all three target columns reach
# 100% cumulative churn, so UPDATE_STREAMS is an upper bound rather than a
# target. 300 streams (~5.1GB) is the verified minimum that drives all three
# target columns to churn_100pct: web_sales gets there by ~200, but both
# catalog_returns columns are still at 80% at 200 streams.

mkdir -p ./data1gbupdates
for i in $(seq 1 "$UPDATE_STREAMS"); do
    if [ ! -f "./data1gbupdates/s_web_order_${i}.dat" ]; then
        ./dsdgen -scale 1 -dir ./data1gbupdates -update "$i" -force
    fi
    if [ $((i % 50)) -eq 0 ]; then
        echo ">>> generated $i / $UPDATE_STREAMS update streams"
    fi
done

# --- 6. Generate queries per template ---
# dsqgen uses -stream N to produce N queries with different random seeds
# sed removes dsqgen metadata lines (name, param, blank lines)

for tpl in custom_ws_item_sk custom_cr_item_sk custom_cr_returned_time_sk; do
    echo ">>> generating queries for $tpl"
    ./dsqgen \
      -template "${tpl}.tpl" \
      -directory ./query_templates \
      -dialect postgres \
      -scale 1 \
      -stream "$WORKLOAD_COUNT" \
      -FILTER | sed '/^name /d; /^param /d; /^[[:space:]]*$/d' \
      > "${tpl}_initial.sql"
done

# --- 7. Remove duplicate queries ---
# dsqgen can produce identical queries for different streams
# This script keeps only unique queries

mkdir -p ./deduped_queries

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

echo "============================================="
echo "DSB setup complete."
echo "  base data    : $(pwd)/data1gb"
echo "  update files : $(pwd)/data1gbupdates"
echo "  queries      : $(pwd)/deduped_queries"
echo ""
echo "Next: export the drift checkpoints with"
echo "  ./setup_env.sh --dsb-drift      (from the repo root)"
echo "============================================="
