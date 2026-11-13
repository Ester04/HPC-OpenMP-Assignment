#!/usr/bin/env bash
# Benchmark the LU sources with perf stat and Valgrind Cachegrind.
# Usage: ./perf_valgrind.sh ["DATASETS=MINI SMALL STANDARD LARGE EXTRALARGE"] \
#        [PERF_REPS=5] ["THREADS=4 8"] \
#        [OUT=lu_results.csv] [VALGRIND_REPS=1]
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
UTIL_DIR="$SCRIPT_DIR/../../../utilities"

DATASETS_TEXT="${1:-MINI SMALL STANDARD LARGE EXTRALARGE}"
PERF_REPS="${2:-5}"
THREADS_TEXT="${3:-4 8}"
OUT="${4:-lu_results.csv}"
VALGRIND_REPS="${5:-1}"
CC="${CC:-gcc}"
CFLAGS="${CFLAGS:--O2 -g}"
OMPFLAGS="${OMPFLAGS:--fopenmp}"
PERF_EVENTS="${PERF_EVENTS:-task-clock,cycles,instructions,cache-references,cache-misses,branch-misses}"

export LC_ALL=C

if [[ ! "$PERF_REPS" =~ ^[1-9][0-9]*$ || ! "$VALGRIND_REPS" =~ ^[1-9][0-9]*$ ]]; then
  echo "Errore: il numero di ripetizioni deve essere un intero positivo." >&2
  exit 2
fi

if ! command -v "$CC" >/dev/null 2>&1; then
  echo "Errore: compilatore '$CC' non trovato." >&2
  exit 2
fi
for tool in perf valgrind awk; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "Errore: '$tool' non trovato nel PATH." >&2
    exit 2
  fi
done

read -r -a DATASETS <<< "$DATASETS_TEXT"
if (("${#DATASETS[@]}" == 0)); then
  echo "Errore: specificare almeno un dataset o una dimensione." >&2
  exit 2
fi

read -r -a THREADS <<< "$THREADS_TEXT"
if (("${#THREADS[@]}" == 0)); then
  echo "Errore: specificare almeno un numero di thread." >&2
  exit 2
fi
for thread_count in "${THREADS[@]}"; do
  if [[ ! "$thread_count" =~ ^[1-9][0-9]*$ ]]; then
    echo "Errore: numero di thread non valido: '$thread_count'." >&2
    exit 2
  fi
done

if [[ "$OUT" != /* ]]; then
  OUT="$PWD/$OUT"
fi
mkdir -p -- "$(dirname -- "$OUT")"

TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/lu-perf-valgrind.XXXXXX")"
trap 'rm -rf -- "$TMP_DIR"' EXIT

read -r -a CFLAGS_ARRAY <<< "$CFLAGS"
read -r -a OMPFLAGS_ARRAY <<< "$OMPFLAGS"
SOURCES=(
  "v0:$SCRIPT_DIR/lu.c"
  "v2:$SCRIPT_DIR/lu_v2.c"
  "v3:$SCRIPT_DIR/lu_v3.c"
  "v4:$SCRIPT_DIR/lu_v4.c"
)
BINS=()
SIZES=()

for dataset in "${DATASETS[@]}"; do
  case "$dataset" in
    STANDARD) size_flag="-DSTANDARD_DATASET=" ;;
    MINI|SMALL|LARGE|EXTRALARGE) size_flag="-D${dataset}_DATASET" ;;
    *)
      if [[ ! "$dataset" =~ ^[1-9][0-9]*$ ]]; then
        echo "Errore: dataset o dimensione non validi: '$dataset'." >&2
        exit 2
      fi
      size_flag="-DN=$dataset"
      ;;
  esac
  N="$(printf '#include "lu.h"\nN\n' | \
    "$CC" "${CFLAGS_ARRAY[@]}" "${OMPFLAGS_ARRAY[@]}" "$size_flag" \
      -I"$SCRIPT_DIR" -E -P -x c -)"
  if [[ ! "$N" =~ ^[1-9][0-9]*$ ]]; then
    echo "Errore: dimensione non valida per '$dataset': '$N'." >&2
    exit 2
  fi
  for previous in ${SIZES[@]+"${SIZES[@]}"}; do
    if [[ "$N" == "$previous" ]]; then
      echo "Errore: dimensione N=$N ripetuta; selezionare dataset distinti." >&2
      exit 2
    fi
  done
  SIZES+=( "$N" )

  for entry in "${SOURCES[@]}"; do
    version="${entry%%:*}"
    source_file="${entry#*:}"
    binary="$TMP_DIR/lu_${version}_n${N}"
    echo "Compilo $version, dataset=$dataset, N=$N..." >&2
    "$CC" "${CFLAGS_ARRAY[@]}" "${OMPFLAGS_ARRAY[@]}" "$size_flag" \
      -I"$UTIL_DIR" "$source_file" "$UTIL_DIR/polybench.c" -lm -o "$binary"
    BINS+=( "$version:$N:$binary" )
  done
done

echo "tool,version,n,threads,rep,wall_seconds,task_clock_ms,cycles,instructions,cache_references,cache_misses,branch_misses,vg_ir,vg_i1mr,vg_ilmr,vg_dr,vg_d1mr,vg_dlmr,vg_dw,vg_d1mw,vg_dlmw,exit_status" > "$OUT"

perf_values() {
  local perf_file="$1"
  awk -F, '
    {
      event = $3
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", event)
      value = $1
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", value)
      gsub(/,/, "", value)
      if (value ~ /</ || value == "n/a") value = ""
      if (event == "task-clock") task_clock = value
      else if (event == "cycles") cycles = value
      else if (event == "instructions") instructions = value
      else if (event == "cache-references") cache_references = value
      else if (event == "cache-misses") cache_misses = value
      else if (event == "branch-misses") branch_misses = value
    }
    END {
      printf "%s,%s,%s,%s,%s,%s\n", task_clock, cycles, instructions, \
        cache_references, cache_misses, branch_misses
    }
  ' "$perf_file"
}

cachegrind_values() {
  local cache_file="$1"
  awk '
    $1 == "events:" {
      for (i = 2; i <= NF; i++) event_index[$i] = i - 1
    }
    $1 == "summary:" {
      wanted[1] = "Ir"
      wanted[2] = "I1mr"
      wanted[3] = "ILmr"
      wanted[4] = "Dr"
      wanted[5] = "D1mr"
      wanted[6] = "DLmr"
      wanted[7] = "Dw"
      wanted[8] = "D1mw"
      wanted[9] = "DLmw"
      for (i = 1; i <= 9; i++) {
        event_col = event_index[wanted[i]]
        result[i] = event_col ? $(event_col + 1) : ""
      }
    }
    END {
      for (i = 1; i <= 9; i++) {
        gsub(/,/, "", result[i])
        printf "%s%s", result[i], (i == 9 ? "\n" : ",")
      }
    }
  ' "$cache_file"
}

run_timed() {
  local time_file="$1"
  shift
  local status
  set +e
  { TIMEFORMAT='%R'; time "$@" >/dev/null 2>&1; } 2> "$time_file"
  status=$?
  set -e
  printf '%s' "$status"
}

echo "Misure perf..."
for item in "${BINS[@]}"; do
  version="${item%%:*}"
  remaining="${item#*:}"
  N="${remaining%%:*}"
  binary="${remaining#*:}"
  for thread_count in "${THREADS[@]}"; do
    for ((rep = 1; rep <= PERF_REPS; rep++)); do
      perf_file="$TMP_DIR/perf_${version}_${N}_${thread_count}_${rep}.csv"
      time_file="$TMP_DIR/time_${version}_${N}_${thread_count}_${rep}.txt"
      echo "perf $version, N=$N, thread=$thread_count, ripetizione=$rep" >&2
      status="$(run_timed "$time_file" env OMP_NUM_THREADS="$thread_count" \
        perf stat -x, -o "$perf_file" -e "$PERF_EVENTS" -- "$binary")"
      wall_seconds="$(awk 'NF { print $1; exit }' "$time_file")"
      if [[ -f "$perf_file" ]]; then
        values="$(perf_values "$perf_file")"
      else
        values=",,,,,"
      fi
      printf 'perf,%s,%s,%s,%s,%s,%s,,,,,,,,,,%s\n' \
        "$version" "$N" "$thread_count" "$rep" "$wall_seconds" "$values" "$status" >> "$OUT"
      if [[ "$status" != 0 ]]; then
        echo "Avviso: perf non ha completato correttamente (exit $status)." >&2
      fi
    done
  done
done

echo "Misure Valgrind Cachegrind..."
for item in "${BINS[@]}"; do
  version="${item%%:*}"
  remaining="${item#*:}"
  N="${remaining%%:*}"
  binary="${remaining#*:}"
  for thread_count in "${THREADS[@]}"; do
    for ((rep = 1; rep <= VALGRIND_REPS; rep++)); do
      cache_file="$TMP_DIR/cache_${version}_${N}_${thread_count}_${rep}.out"
      log_file="$TMP_DIR/valgrind_${version}_${N}_${thread_count}_${rep}.log"
      time_file="$TMP_DIR/time_vg_${version}_${N}_${thread_count}_${rep}.txt"
      echo "valgrind $version, N=$N, thread=$thread_count, ripetizione=$rep" >&2
      status="$(run_timed "$time_file" env OMP_NUM_THREADS="$thread_count" \
        valgrind --tool=cachegrind --cachegrind-out-file="$cache_file" \
        --log-file="$log_file" "$binary")"
      wall_seconds="$(awk 'NF { print $1; exit }' "$time_file")"
      values=",,,,,,,,"
      if [[ -f "$cache_file" ]]; then
        values="$(cachegrind_values "$cache_file")"
      else
        echo "Avviso: Cachegrind non ha prodotto i contatori per $version." >&2
      fi
      printf 'valgrind,%s,%s,%s,%s,%s,,,,,,,%s,%s\n' \
        "$version" "$N" "$thread_count" "$rep" "$wall_seconds" "$values" "$status" >> "$OUT"
      if [[ "$status" != 0 ]]; then
        echo "Avviso: Valgrind non ha completato correttamente (exit $status)." >&2
      fi
    done
  done
done

echo "Risultati salvati in: $OUT"


#USED on Nano1:
#OMP_DYNAMIC=FALSE OMP_PROC_BIND=TRUE OMP_PLACES=cores \
#  ./perf_valgrind.sh

#python3 plot_lu_results.py lu_results.csv