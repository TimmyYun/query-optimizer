#!/usr/bin/env bash
# SCALE=1 UPDATE_STREAMS=300 bash "DSB benchmark/setup.sh"
set -euo pipefail

SCALE="${SCALE:-1}"
UPDATE_STREAMS="${UPDATE_STREAMS:-300}"
WORKLOAD_COUNT="${WORKLOAD_COUNT:-400000}"
CHURN_STEP="${CHURN_STEP:-5}"
MAX_CHURN="${MAX_CHURN:-100}"
ITEM_HISTORY="${ITEM_HISTORY:-0}"
SAMPLE_SIZE="${SAMPLE_SIZE:-100000}"
QUERIES_PER_STEP="${QUERIES_PER_STEP:-200000}"
SEED="${SEED:-42}"
RANK="${RANK:-1}"
COLUMNS="${COLUMNS:-store_sales.ss_item_sk catalog_sales.cs_item_sk catalog_returns.cr_item_sk web_sales.ws_item_sk catalog_returns.cr_returned_time_sk}"
DSB_COMMIT="ec9a156ceee923db1114cbe388f6183b53d49787"
RNGSEED=19620718
JOBS="${JOBS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu)}"
PYTHON="${PYTHON:-poetry run python}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
WORK="${WORK:-$HERE/work}"
TOOLS="$WORK/dsb/code/tools"
BASE_REL="sf$SCALE/base"
UPDATES_REL="sf$SCALE/updates"
BASE="$TOOLS/$BASE_REL"
UPDATES="$TOOLS/$UPDATES_REL"
QUERIES="$TOOLS/sf$SCALE/queries/n$WORKLOAD_COUNT"
OUT="${OUT:-$ROOT/data/dsb/sf$SCALE}"

echo ">>> DSB $DSB_COMMIT"
if [ ! -d "$WORK/dsb/.git" ]; then
    mkdir -p "$WORK/dsb"
    git -C "$WORK/dsb" init -q
    git -C "$WORK/dsb" fetch -q --depth 1 https://github.com/microsoft/dsb.git "$DSB_COMMIT"
    git -C "$WORK/dsb" checkout -q FETCH_HEAD
fi
[ "$(git -C "$WORK/dsb" rev-parse HEAD)" = "$DSB_COMMIT" ] || { echo "dsb не на коммите $DSB_COMMIT"; exit 1; }

echo ">>> build"
if [ ! -x "$TOOLS/dsdgen" ] || [ ! -x "$TOOLS/dsqgen" ]; then
    (
        cd "$TOOLS"
        if [ "$(uname)" = "Darwin" ]; then
            sed -i '' 's|^#define USE_VALUES_H|#define USE_LIMITS_H|' config.h
            grep -rl "include <malloc.h>" --include="*.c" . | grep -v y.tab.c | xargs sed -i '' 's|#include <malloc.h>|#include <stdlib.h>|' || true
            make OS=LINUX LINUX_CFLAGS="-g -O2 -DMAXINT=2147483647 -Wno-implicit-function-declaration -Wno-int-conversion -Wno-implicit-int -Wno-return-type -Wno-deprecated-non-prototype -Wno-format"
        else
            make OS=LINUX
        fi
    )
fi

echo ">>> base data, scale $SCALE"
if [ ! -f "$BASE/.done" ]; then
    mkdir -p "$BASE"
    (cd "$TOOLS" && ./dsdgen -scale "$SCALE" -dir "$BASE_REL" -rngseed "$RNGSEED" -force)
    touch "$BASE/.done"
fi

echo ">>> $UPDATE_STREAMS update streams, $JOBS jobs"
mkdir -p "$UPDATES"
gen_update() {
    [ -f "$UPDATES/.stream_$1" ] && return 0
    (cd "$TOOLS" && ./dsdgen -scale "$SCALE" -dir "$UPDATES_REL" -update "$1" -rngseed "$RNGSEED" -force > /dev/null)
    touch "$UPDATES/.stream_$1"
}
export -f gen_update
export TOOLS UPDATES UPDATES_REL SCALE RNGSEED
seq 1 "$UPDATE_STREAMS" | xargs -P "$JOBS" -I{} bash -c 'gen_update {}'

echo ">>> $WORKLOAD_COUNT DSB queries per column"
mkdir -p "$TOOLS/query_templates" "$QUERIES"
cp "$WORK/dsb/query_templates_pg/spj_queries/postgres.tpl" "$TOOLS/query_templates/postgres.tpl"
ITEM_MAX="$(cut -d'|' -f1 "$BASE/item.dat" | sort -n | tail -1)"
for name in $COLUMNS; do
    col="${name#*.}"
    case "$col" in
        cr_item_sk) lo=1; hi=$((ITEM_MAX - 1)); wlo=700; whi=3000 ;;
        ws_item_sk) lo=1; hi=$((ITEM_MAX - 1)); wlo=400; whi=2000 ;;
        *_item_sk) lo=1; hi=$((ITEM_MAX - 1)); wlo=400; whi=3000 ;;
        *_time_sk) lo=2; hi=86397; wlo=1000; whi=4000 ;;
        *) echo "нет шаблона запросов для $name"; continue ;;
    esac
    [ -s "$QUERIES/$name.sql" ] && continue
    printf 'define START = random(%d, %d, uniform);\ndefine RANGE_ = random(%d, %d, uniform);\n\nWHERE %s BETWEEN [START] AND [START] + [RANGE_];\n' \
        "$lo" "$hi" "$wlo" "$whi" "$col" > "$TOOLS/query_templates/q_$col.tpl"
    (cd "$TOOLS" && ./dsqgen -template "q_$col.tpl" -directory query_templates -dialect postgres \
        -scale "$SCALE" -stream "$WORKLOAD_COUNT" -rngseed "$((RNGSEED + $(printf '%s' "$name" | cksum | cut -d' ' -f1) % 100000))" -FILTER) \
        | sed '/^name /d; /^param /d; /^[[:space:]]*$/d' > "$QUERIES/$name.sql.tmp"
    mv "$QUERIES/$name.sql.tmp" "$QUERIES/$name.sql"
done

cd "$ROOT"
if [ "$RANK" = "1" ]; then
    echo ">>> rank columns"
    $PYTHON "$HERE/dsb_drift.py" rank --tools "$TOOLS" --base "$BASE" --out "$OUT/rank.csv" --seed "$SEED"
fi

echo ">>> drift checkpoints"
HISTORY_FLAG=()
[ "$ITEM_HISTORY" = "1" ] && HISTORY_FLAG=(--item-history)
$PYTHON "$HERE/dsb_drift.py" drift --tools "$TOOLS" --base "$BASE" --updates "$UPDATES" --queries "$QUERIES" \
    --streams "$UPDATE_STREAMS" --columns $COLUMNS --churn-step "$CHURN_STEP" --max-churn "$MAX_CHURN" \
    --sample-size "$SAMPLE_SIZE" --queries-per-step "$QUERIES_PER_STEP" --seed "$SEED" ${HISTORY_FLAG[@]+"${HISTORY_FLAG[@]}"} \
    --out "$OUT"
echo ">>> done: $OUT"
