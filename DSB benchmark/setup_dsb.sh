#!/usr/bin/env bash
# ============================================================
# DSB Benchmark: build the official toolkit, generate data + refresh streams
# Linux and macOS (Apple clang).
#
# Everything comes from microsoft/dsb, pinned to DSB_COMMIT, unmodified
# apart from the portability patches below. No custom query templates:
# queries are generated later from the official query_templates_pg by
#   python prepare_benchmark.py --bench dsb --generate-queries
#
# setup.sh is the original Linux script and is left untouched for reference.
#
# Safe to re-run: clone, build and generation steps are skipped
# if their output already exists.
# ============================================================
set -e

# --- 0. CONFIGURATION (all of it is reported in README.md) ---
DSB_COMMIT=${DSB_COMMIT:-ec9a156ceee923db1114cbe388f6183b53d49787}   # microsoft/dsb master, 2024-11-07
SCALE=${SCALE:-1}                    # scale factor (GB)
UPDATE_STREAMS=${UPDATE_STREAMS:-300}
# dsdgen uses its built-in default RNG seed; data is deterministic for a given SCALE.

# Always work relative to this script, not the caller's cwd
cd "$(dirname "$0")"

echo "============================================="
echo "DSB setup ($(uname -s))"
echo "Commit         : $DSB_COMMIT"
echo "Scale factor   : $SCALE"
echo "Update streams : $UPDATE_STREAMS"
echo "============================================="

# --- 1. Clone DSB repository at a fixed commit ---
if [ ! -d "./dsb" ]; then
    git clone https://github.com/microsoft/dsb.git
fi
git -C ./dsb checkout -q "$DSB_COMMIT"

cd ./dsb/code/tools

# --- 2. Portability patches ---
# GCC >= 10 and clang >= 11 default to -fno-common, and GCC 14 / clang 15
# turned implicit declarations and int conversions into errors; this
# K&R-era C trips all of them. The flags restore the old behaviour and
# do not change generated data.
# Ubuntu/Debian GCC also enables _FORTIFY_SOURCE at -O2 by default; its
# conservative size checks abort dsdgen ("*** buffer overflow detected ***")
# on the toolkit's fixed-size strcpy/sprintf buffers. Fortify only aborts,
# it computes nothing, so turning it off leaves the data unchanged.
CFLAGS_COMPAT="-g -O2 -U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=0 -fcommon -Wno-implicit-function-declaration -Wno-int-conversion -Wno-implicit-int -Wno-return-type -Wno-format -Wno-incompatible-pointer-types"

if [ "$(uname -s)" = "Darwin" ]; then
    # macOS only:
    #   1. <values.h> does not exist. Its only symbol used here is MAXINT,
    #      which glibc defines as INT_MAX -- so switch the LINUX block to
    #      <limits.h> and define MAXINT identically. Using the same value
    #      matters: genrand.c does arithmetic mod MAXINT, so any other
    #      value would change every generated row.
    #   2. <malloc.h> does not exist; malloc() lives in <stdlib.h>.
    # Both are idempotent.
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
    CFLAGS_COMPAT="$CFLAGS_COMPAT -Wno-deprecated-non-prototype"
fi

# --- 3. Build dsdgen and dsqgen ---
# Rebuild whenever the flags differ from the last build. Not "make clean":
# it also deletes the pre-generated y.tab.c / y.tab.h / tokenizer.c that the
# repo ships, and regenerating them would need yacc + lex.
if [ "$(cat .build_flags 2>/dev/null)" != "$CFLAGS_COMPAT" ]; then
    echo ">>> compiler flags changed, rebuilding"
    rm -f ./*.o ./dsdgen ./dsqgen ./distcomp ./mkheader ./checksum ./tpcds.idx
fi
# Restore the shipped parser sources if an earlier "make clean" removed them
git checkout -- y.tab.c y.tab.h tokenizer.c 2>/dev/null || true

if [ ! -x ./dsdgen ] || [ ! -x ./dsqgen ]; then
    if make OS=LINUX LINUX_CFLAGS="$CFLAGS_COMPAT"; then
        echo "$CFLAGS_COMPAT" > .build_flags
    else
        echo ""
        echo "ERROR: build failed. Check the compiler output above."
        echo "  - missing <values.h> / <malloc.h> on macOS -> see the patches above"
        echo "  - new implicit-declaration / conversion errors -> add -Wno-... to CFLAGS_COMPAT"
        exit 1
    fi
else
    echo ">>> dsdgen and dsqgen already built, skipping make"
fi

# --- 4. Generate base data ---
# ~23 .dat files in ./data1gb (the folder name is kept for SCALE=1 compatibility)
mkdir -p ./data1gb
if [ ! -f ./data1gb/web_sales.dat ]; then
    ./dsdgen -scale "$SCALE" -dir ./data1gb -force
else
    echo ">>> ./data1gb already populated, skipping"
fi

# --- 5. Generate refresh streams ---
# Each stream produces ~23 staging files (s_*.dat, delete_*.dat), ~17 MB at SF=1.
# The notebook stops as soon as every target column reaches 100% cumulative
# churn, so UPDATE_STREAMS is an upper bound. Measured at SF=1: web_sales
# reaches 100% churn by ~200 streams; 300 leaves a margin.
mkdir -p ./data1gbupdates
for i in $(seq 1 "$UPDATE_STREAMS"); do
    if [ ! -f "./data1gbupdates/s_web_order_${i}.dat" ]; then
        ./dsdgen -scale "$SCALE" -dir ./data1gbupdates -update "$i" -force
    fi
    if [ $((i % 50)) -eq 0 ]; then
        echo ">>> generated $i / $UPDATE_STREAMS update streams"
    fi
done

echo "============================================="
echo "DSB setup complete."
echo "  base data    : $(pwd)/data1gb"
echo "  update files : $(pwd)/data1gbupdates"
echo ""
echo "Next (from the repo root):"
echo "  ./setup_env.sh --dsb"
echo "============================================="
