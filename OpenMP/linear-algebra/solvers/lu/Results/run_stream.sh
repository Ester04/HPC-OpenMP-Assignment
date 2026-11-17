#!/usr/bin/env bash
# Misura la banda di memoria (STREAM) con 1 e 4 thread e la confronta con quella
# richiesta da LU v0/v3 (GFLOPS/s del CSV lu_memory-bound.csv).
# Usage: ./run_stream.sh ["THREADS=1 4"] [ARRAY_SIZE=20000000] [OUT=stream_results.txt]
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
THREADS_TEXT="${1:-1 4}"
ARRAY_SIZE="${2:-20000000}"
OUT="${3:-stream_results.txt}"
CC="${CC:-gcc}"
CFLAGS="${CFLAGS:--O2}"
OMPFLAGS="${OMPFLAGS--fopenmp}"
RESULTS_CSV="${RESULTS_CSV:-$SCRIPT_DIR/lu_memory-bound.csv}"

export LC_ALL=C
read -r -a THREADS <<< "$THREADS_TEXT"
read -r -a CFLAGS_ARRAY <<< "$CFLAGS"
read -r -a OMPFLAGS_ARRAY <<< "$OMPFLAGS"
[[ "$OUT" == /* ]] || OUT="$PWD/$OUT"

TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/stream.XXXXXX")"
trap 'rm -rf -- "$TMP_DIR"' EXIT
BIN="$TMP_DIR/stream_bw"

"$CC" "${CFLAGS_ARRAY[@]}" ${OMPFLAGS_ARRAY[@]+"${OMPFLAGS_ARRAY[@]}"} -DSTREAM_ARRAY_SIZE="${ARRAY_SIZE}L" \
  "$SCRIPT_DIR/stream_bw.c" -o "$BIN"

: > "$OUT"
for t in "${THREADS[@]}"; do
  echo "STREAM con $t thread..." >&2
  OMP_NUM_THREADS="$t" OMP_DYNAMIC=FALSE OMP_PROC_BIND=true "$BIN" | tee -a "$OUT"
  echo >> "$OUT"
done

# Confronto con LU: banda richiesta = GFLOPS * (byte/FLOP).
# Update LU: legge A[i][j], A[k][j] e riscrive A[i][j] -> 24 B per 2 FLOP (12 B/FLOP) senza riuso;
# con A[k][*] in cache: 16 B per 2 FLOP (8 B/FLOP).
if [[ -f "$RESULTS_CSV" ]]; then
  {
    echo "== Banda richiesta da LU (mediana GFLOPS/s del CSV) =="
    printf '%-4s %-6s %-4s %10s %14s %14s\n' ver N thr GFLOPS "BW@8B/FLOP" "BW@12B/FLOP"
    # mawk (default su Ubuntu 18.04) non ha asort: la mediana si calcola con sort.
    awk -F, 'NR > 1 { print $1 "," $2 "," $3 }' "$RESULTS_CSV" | sort -u -t, -k2,2n -k1,1 | \
    while IFS=, read -r ver n thr; do
      med="$(awk -F, -v v="$ver" -v n="$n" -v t="$thr" '$1==v && $2==n && $3==t { print $6 }' "$RESULTS_CSV" | sort -n | \
        awk '{ x[NR] = $1 } END { print x[int((NR + 1) / 2)] }')"
      awk -v v="$ver" -v n="$n" -v t="$thr" -v g="$med" \
        'BEGIN { printf "%-4s %-6s %-4s %10.3f %11.2f GB/s %11.2f GB/s\n", v, n, t, g, g * 8, g * 12 }'
    done
  } 2>/dev/null | tee -a "$OUT"
fi
echo "Risultati in: $OUT" >&2
