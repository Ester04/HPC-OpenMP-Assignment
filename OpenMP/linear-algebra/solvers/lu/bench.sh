#!/usr/bin/env bash
# Uso: ./bench.sh [N=2000] [REPS=5] ["THREADS"="1 2 4 8"] [OUT=results.csv]
# Esegue tutte le versioni e stampa tempo (min/mediana), GFLOP/s, speedup ed
# efficienza rispetto alla baseline seriale (versione 0).
set -euo pipefail
cd "$(dirname "$0")"

N=${1:-2000}
REPS=${2:-5}
THREADS=${3:-"1 2 4 8"}
OUT=${4:-results.csv}

make -s -f Makefile.omp all

echo "version,n,threads,min_s,median_s,gflops,maxreldiff,residual" > "$OUT"

# Baseline seriale
./lu_v0 "$N" "$REPS" 0 >> "$OUT"

for v in 1 2 3 4; do
  for t in $THREADS; do
    OMP_NUM_THREADS=$t ./lu_v$v "$N" "$REPS" 0 >> "$OUT"
  done
done

echo
echo "N=$N  REPS=$REPS  (tempi in secondi; speedup rispetto a v0 seriale, su min)"
awk -F, '
  NR == 1 { printf "%-8s %-8s %10s %10s %9s %9s %8s\n", "version","threads","min_s","median_s","GFLOP/s","speedup","eff"; next }
  NR == 2 { base = $4 }
  { s = base / $4; printf "%-8s %-8s %10s %10s %9s %9.2f %8.2f\n", $1, $3, $4, $5, $6, s, s / $3 }
' "$OUT"
echo
echo "Dati grezzi: $OUT"