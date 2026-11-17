#!/usr/bin/env bash
# sysinfo.sh - raccoglie CPU, core fisici, cache, flag di compilazione e numero di prove.
# Uso:  ./sysinfo.sh [BINARIO] [FILE_CSV] > sysinfo.txt
#   BINARIO  (facoltativo) eseguibile del kernel, per controllare istruzioni vettoriali e flag
#   FILE_CSV (facoltativo) CSV dei risultati, per contare le prove per combinazione
# Esegui il comando nella cartella del Makefile, sulla stessa macchina delle misure.
BIN="${1:-}"
CSV="${2:-}"
CC_BIN="${CC:-gcc}"

sec() { printf '\n== %s ==\n' "$1"; }

sec "Data e host"
date; hostname; uname -sr

sec "CPU"
lscpu | grep -E '^(Model name|Architecture|CPU\(s\)|Thread\(s\) per core|Core\(s\) per socket|Socket\(s\)|NUMA node\(s\)|CPU max MHz|CPU min MHz|CPU MHz)' \
  || grep -m1 'model name' /proc/cpuinfo
echo "Istruzioni vettoriali presenti:"
grep -o -w -E 'sse4_2|avx|avx2|fma|avx512f' /proc/cpuinfo | sort -u | tr '\n' ' '; echo

sec "Core fisici contro thread logici (stessa colonna CORE = thread SMT sullo stesso core)"
lscpu -e=CPU,CORE,SOCKET,ONLINE 2>/dev/null | head -40 || echo "lscpu -e non disponibile"
PHYS=$(lscpu -p=CORE,SOCKET 2>/dev/null | grep -v '^#' | sort -u | wc -l)
LOGI=$(nproc --all)
echo "Core fisici: $PHYS   Thread logici: $LOGI   (CPU disponibili al processo: $(env -u OMP_NUM_THREADS nproc))"

sec "Cache"
lscpu | grep -E '^(L1d|L1i|L2|L3)' || getconf -a 2>/dev/null | grep -i CACHE

sec "Frequenza e turbo"
G=/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
[ -r "$G" ] && echo "governor: $(cat $G)" || echo "governor: non leggibile"
NT=/sys/devices/system/cpu/intel_pstate/no_turbo
[ -r "$NT" ] && echo "intel no_turbo: $(cat $NT)  (1 = turbo spento)"

sec "Limiti e carico (macchina condivisa?)"
echo "CPU consentite a questa shell: $(taskset -cp $$ 2>/dev/null | sed 's/.*: //')"
[ -r /sys/fs/cgroup/cpu.max ] && echo "cgroup cpu.max: $(cat /sys/fs/cgroup/cpu.max)  (max = nessun limite)"
uptime

sec "Compilatore"
$CC_BIN --version | head -1
echo "Flag usati dal Makefile (make -n, senza eseguire):"
make -n 2>/dev/null | grep -E "(^|[ /])(gcc|cc|clang)[ ]" | head -3 || echo "  nessun Makefile qui"

sec "Thread OpenMP effettivi (programma di prova)"
TMP=$(mktemp -d)
cat > "$TMP/t.c" <<'EOF'
#define _GNU_SOURCE
#include <omp.h>
#include <sched.h>
#include <stdio.h>
int main(void) {
  printf("omp_get_max_threads=%d  omp_get_num_procs=%d\n", omp_get_max_threads(), omp_get_num_procs());
  #pragma omp parallel
  {
    #pragma omp critical
    printf("  thread %d su CPU %d (team di %d)\n", omp_get_thread_num(), sched_getcpu(), omp_get_num_threads());
  }
  return 0;
}
EOF
if $CC_BIN -fopenmp -o "$TMP/t" "$TMP/t.c" 2>/dev/null; then
  echo "OMP_NUM_THREADS=${OMP_NUM_THREADS:-non impostato}  OMP_PROC_BIND=${OMP_PROC_BIND:-non impostato}  OMP_PLACES=${OMP_PLACES:-non impostato}"
  OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" "$TMP/t"
else
  echo "compilazione OpenMP di prova fallita"
fi
rm -rf "$TMP"

if [ -n "$BIN" ] && [ -x "$BIN" ]; then
  sec "Binario: $BIN"
  echo "Istruzioni vettoriali nel codice (ymm = AVX 256 bit, zmm = AVX-512): ymm=$(objdump -d "$BIN" | grep -c ymm)  zmm=$(objdump -d "$BIN" | grep -c zmm)"
  echo "FMA (vfmadd): $(objdump -d "$BIN" | grep -c vfmadd)"
  echo "Flag registrati nel binario (solo se compilato con -frecord-gcc-switches):"
  readelf -p .GCC.command.line "$BIN" 2>/dev/null | head -5 || true
  readelf --debug-dump=info "$BIN" 2>/dev/null | grep -m1 DW_AT_producer || echo "  nessuna informazione (ricompila con -g o -frecord-gcc-switches)"
fi

if [ -n "$CSV" ] && [ -r "$CSV" ]; then
  sec "Prove nel CSV: $CSV"
  echo "Intestazione:"; head -1 "$CSV"
  echo "Righe di dati: $(($(wc -l < "$CSV") - 1))"
  echo "Adatta le colonne e lancia, per esempio:"
  echo "  python3 -c \"import pandas as pd; d=pd.read_csv('$CSV'); print(d.groupby(['version','N','threads']).size().describe())\""
fi