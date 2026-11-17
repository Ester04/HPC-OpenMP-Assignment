#!/usr/bin/env bash
# Confronta LU v0 (seriale originale, lu.c) e v3 (lu_v3.c) al variare di N misurando
# con perf: GFLOPS/s, backend bound e (se la CPU espone le metriche) memory bound.
# Usage: ./perf_v0_v3_topdown.sh ["SIZES=32 128 1024 2000"] [REPS=5] ["THREADS=4"] [OUT=lu_v0_v3_topdown.csv]
#   Variabili: CC, CFLAGS, OMPFLAGS, KEEP_LOGS=1 (conserva i log grezzi di perf in <OUT>_logs/)
# GFLOPS/s = (2/3 * N^3) / kernel_seconds, con kernel_seconds stampato da PolyBench.
# v0 e' sempre eseguita con 1 thread; v3 con THREADS (puo' essere una lista).
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
UTIL_DIR="$SCRIPT_DIR/../../../../utilities"
LU_DIR="$SCRIPT_DIR/.."

SIZES_TEXT="${1:-32 128 1024 2000}"
REPS="${2:-5}" # {2:-5} = default value 5 if not provided as second argument
THREADS_TEXT="${3:-4}"
OUT="${4:-lu_v0_v3_topdown.csv}"
CC="${CC:-gcc}"
CFLAGS="${CFLAGS:--O2 -g}"
OMPFLAGS="${OMPFLAGS:--fopenmp}"
KEEP_LOGS="${KEEP_LOGS:-0}"

export LC_ALL=C

if [[ ! "$REPS" =~ ^[1-9][0-9]*$ ]]; then
  echo "Errore: REPS deve essere un intero positivo." >&2
  exit 2
fi
for tool in perf awk "$CC"; do
  command -v "$tool" >/dev/null 2>&1 || { echo "Errore: '$tool' non trovato nel PATH." >&2; exit 2; }
done

read -r -a SIZES <<< "$SIZES_TEXT"
read -r -a THREADS <<< "$THREADS_TEXT"
read -r -a CFLAGS_ARRAY <<< "$CFLAGS"
read -r -a OMPFLAGS_ARRAY <<< "$OMPFLAGS"

[[ "$OUT" == /* ]] || OUT="$PWD/$OUT"
if [[ -e "$OUT" ]]; then
  echo "Errore: il file di output esiste gia': $OUT" >&2
  exit 2
fi
mkdir -p -- "$(dirname -- "$OUT")"

TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/lu-v0-v3.XXXXXX")"
trap 'rm -rf -- "$TMP_DIR"' EXIT
LOG_DIR="${OUT%.csv}_logs"
[[ "$KEEP_LOGS" == 1 ]] && mkdir -p "$LOG_DIR"

BASIC_EVENTS="task-clock,cycles,instructions,cache-references,cache-misses,stalled-cycles-frontend,stalled-cycles-backend"

# Prova in ordine diversi gruppi di metriche (Intel: TopdownL1/L2, AMD: PipelineL1/L2).
# Il primo che produce output valido viene usato.
L1_METRICS=("TopdownL1" "PipelineL1" "tma_backend_bound")
L2_METRICS=("TopdownL2" "PipelineL2" "tma_memory_bound")

# Estrae da un log di perf il valore (%) che precede il nome di metrica indicato.
metric_pct() {
  local file="$1" pattern="$2"
  awk -v pat="$pattern" '
    BEGIN { IGNORECASE = 1 }
    $0 ~ pat && /%/ {
      for (i = 1; i <= NF; i++) if ($i == "%" || $i ~ /%$/) {
        v = ($i == "%") ? $(i-1) : $i
        gsub(/%/, "", v)
        if (v ~ /^[0-9.]+$/) { print v; exit }
      }
    }' "$file"
}

# Valore di un evento in un log perf stat (testo): primo campo numerico della riga con quel nome.
event_value() {
  local file="$1" event="$2"
  awk -v ev="$event" '
    $0 ~ ("[[:space:]]" ev "([[:space:]]|$)") {
      v = $1; gsub(/,/, "", v)
      if (v ~ /^[0-9.]+$/) { print v; exit }
    }' "$file"
}

# Esegue perf con la prima lista di metriche funzionante; stampa il nome usato (o nulla).
run_metric_group() {
  local log="$1" threads="$2" binary="$3"; shift 3
  local metric
  for metric in "$@"; do
    if env OMP_NUM_THREADS="$threads" OMP_DYNAMIC=FALSE \
         perf stat -M "$metric" -o "$log" -- "$binary" >/dev/null 2>&1 \
       && ! grep -qE "not supported|<not counted>|Cannot find" "$log"; then
      echo "$metric"
      return 0
    fi
  done
  : > "$log"
  return 1
}

printf '%s\n' "version,n,threads,rep,kernel_seconds,gflops,wall_seconds,task_clock_ms,cycles,instructions,ipc,cache_references,cache_misses,cache_miss_pct,stalled_frontend_pct,stalled_backend_pct,backend_bound_pct,memory_bound_pct,core_bound_pct,l1_metric,l2_metric" > "$OUT"

for n in "${SIZES[@]}"; do
  if [[ ! "$n" =~ ^[1-9][0-9]*$ ]]; then
    echo "Errore: dimensione non valida: '$n'." >&2
    exit 2
  fi
  for version in v0 v3; do
    if [[ "$version" == v0 ]]; then src="$LU_DIR/lu.c"; else src="$LU_DIR/lu_v3.c"; fi
    binary="$TMP_DIR/lu_${version}_n${n}"
    echo "Compilo $version, N=$n..." >&2
    "$CC" "${CFLAGS_ARRAY[@]}" "${OMPFLAGS_ARRAY[@]}" \
      -UPOLYBENCH_GFLOPS -UPOLYBENCH_CYCLE_ACCURATE_TIMER -UPOLYBENCH_PAPI \
      -DPOLYBENCH_TIME -DPOLYBENCH_NO_FLUSH_CACHE -DN="$n" \
      -I"$UTIL_DIR" "$src" "$UTIL_DIR/polybench.c" -lm -o "$binary"

    if [[ "$version" == v0 ]]; then run_threads=(1); else run_threads=("${THREADS[@]}"); fi
    for t in "${run_threads[@]}"; do
      for ((rep = 1; rep <= REPS; rep++)); do
        echo "  $version N=$n threads=$t rep=$rep" >&2
        tag="${version}_n${n}_t${t}_r${rep}"

        # 1) contatori di base + tempo del kernel (stdout di PolyBench)
        basic="$TMP_DIR/$tag.basic"
        stdout_file="$TMP_DIR/$tag.stdout"
        env OMP_NUM_THREADS="$t" OMP_DYNAMIC=FALSE \
          perf stat -e "$BASIC_EVENTS" -o "$basic" -- "$binary" >"$stdout_file" 2>/dev/null || true
        kernel_s="$(awk 'NF == 1 && $1 ~ /^[0-9.]+$/ { print $1; exit }' "$stdout_file")"
        wall_s="$(awk '/seconds time elapsed/ { print $1; exit }' "$basic")"
        task_ms="$(event_value "$basic" task-clock)"
        cycles="$(event_value "$basic" cycles)"
        instr="$(event_value "$basic" instructions)"
        cref="$(event_value "$basic" cache-references)"
        cmiss="$(event_value "$basic" cache-misses)"
        sfe="$(metric_pct "$basic" "frontend cycles idle")"
        sbe="$(metric_pct "$basic" "backend cycles idle")"
        ipc="$(awk -v i="${instr//,/}" -v c="${cycles//,/}" 'BEGIN { if (c > 0) printf "%.3f", i / c }')"
        cmp="$(awk -v m="${cmiss//,/}" -v r="${cref//,/}" 'BEGIN { if (r > 0) printf "%.2f", 100 * m / r }')"
        gflops="$(awk -v n="$n" -v s="${kernel_s:-0}" 'BEGIN { if (s > 0) printf "%.4f", (2.0 / 3.0) * n * n * n / s / 1e9 }')"

        # 2) top-down livello 1 (backend bound) e 3) livello 2 (memory bound)
        l1log="$TMP_DIR/$tag.l1"; l2log="$TMP_DIR/$tag.l2"
        l1_used="$(run_metric_group "$l1log" "$t" "$binary" "${L1_METRICS[@]}" || true)"
        l2_used="$(run_metric_group "$l2log" "$t" "$binary" "${L2_METRICS[@]}" || true)"
        be_bound="$(metric_pct "$l1log" "backend[_ ]bound")"
        mem_bound="$(metric_pct "$l2log" "memory[_ ]bound")"
        core_bound="$(metric_pct "$l2log" "core[_ ]bound")"
        [[ -z "$be_bound" ]] && be_bound="$(metric_pct "$l2log" "backend[_ ]bound")"

        printf '%s\n' "$version,$n,$t,$rep,${kernel_s:-},${gflops:-},${wall_s:-},${task_ms:-},${cycles//,/},${instr//,/},${ipc:-},${cref//,/},${cmiss//,/},${cmp:-},${sfe:-},${sbe:-},${be_bound:-},${mem_bound:-},${core_bound:-},${l1_used:-n/a},${l2_used:-n/a}" >> "$OUT"

        if [[ "$KEEP_LOGS" == 1 ]]; then
          cat "$basic" "$l1log" "$l2log" > "$LOG_DIR/$tag.perf.txt" 2>/dev/null || true
        fi
      done
    done
  done
done

echo "CSV scritto in: $OUT" >&2

# Riepilogo: media sulle ripetizioni per (versione, N, thread)
awk -F, '
  NR == 1 { next }
  {
    k = $1 "," $2 "," $3
    if (!(k in seen)) { seen[k] = 1; order[++m] = k }
    cnt[k]++
    for (c = 5; c <= 19; c++) if ($c != "") { sum[k, c] += $c; num[k, c]++ }
  }
  function avg(k, c) { return num[k, c] ? sprintf("%.2f", sum[k, c] / num[k, c]) : "n/a" }
  BEGIN { printf "%-4s %-6s %-4s %10s %10s %8s %12s %10s %10s\n", "ver", "N", "thr", "time[s]", "GFLOPS/s", "IPC", "stallBE[%]", "BE_bound%", "Mem_bound%" }
  END {
    for (i = 1; i <= m; i++) {
      k = order[i]; split(k, p, ",")
      printf "%-4s %-6s %-4s %10s %10s %8s %12s %10s %10s\n", p[1], p[2], p[3], avg(k, 5), avg(k, 6), avg(k, 11), avg(k, 16), avg(k, 17), avg(k, 18)
    }
  }' "$OUT"
