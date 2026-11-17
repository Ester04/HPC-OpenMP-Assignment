/* Benchmark in stile STREAM (Copy, Scale, Add, Triad) piu' un kernel "Update"
   identico all'aggiornamento interno della LU: a[j] -= s * b[j].
   Misura la banda di memoria sostenibile; si riporta il tempo MIGLIORE
   (come nello STREAM originale), scartando la prima iterazione.
   Compilazione: gcc -O2 -fopenmp -DSTREAM_ARRAY_SIZE=20000000 stream_bw.c -o stream_bw */
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <sys/time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef STREAM_ARRAY_SIZE
#define STREAM_ARRAY_SIZE 20000000L /* 160 MB per array: >> L2 (2 MB) */
#endif
#ifndef NTIMES
#define NTIMES 20
#endif

#define NKERN 5

static double now(void)
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + 1e-6 * tv.tv_usec;
}

int main(void)
{
  const long n = STREAM_ARRAY_SIZE;
  double *a = aligned_alloc(64, n * sizeof(double));
  double *b = aligned_alloc(64, n * sizeof(double));
  double *c = aligned_alloc(64, n * sizeof(double));
  if (!a || !b || !c) { fprintf(stderr, "allocazione fallita\n"); return 1; }

  /* First-touch parallelo: le pagine vengono assegnate nello stesso modo dei kernel. */
  #pragma omp parallel for schedule(static)
  for (long j = 0; j < n; j++) { a[j] = 1.0; b[j] = 2.0; c[j] = 0.0; }

  const char *name[NKERN] = {"Copy", "Scale", "Add", "Triad", "LU-update"};
  /* Byte per elemento, senza contare il write-allocate. */
  const double bytes[NKERN] = {16, 16, 24, 24, 24};
  double best[NKERN], sum[NKERN];
  for (int k = 0; k < NKERN; k++) { best[k] = DBL_MAX; sum[k] = 0; }
  const double s = 3.0;
  const double s_small = 1e-9; /* evita overflow nel kernel LU-update */

  int threads = 1;
  #ifdef _OPENMP
  threads = omp_get_max_threads();
  #endif

  for (int it = 0; it < NTIMES; it++) {
    double t[NKERN];

    t[0] = now();
    #pragma omp parallel for schedule(static)
    for (long j = 0; j < n; j++) c[j] = a[j];
    t[0] = now() - t[0];

    t[1] = now();
    #pragma omp parallel for schedule(static)
    for (long j = 0; j < n; j++) b[j] = s * c[j];
    t[1] = now() - t[1];

    t[2] = now();
    #pragma omp parallel for schedule(static)
    for (long j = 0; j < n; j++) c[j] = a[j] + b[j];
    t[2] = now() - t[2];

    t[3] = now();
    #pragma omp parallel for schedule(static)
    for (long j = 0; j < n; j++) a[j] = b[j] + s * c[j];
    t[3] = now() - t[3];

    t[4] = now();
    #pragma omp parallel for simd schedule(static)
    for (long j = 0; j < n; j++) a[j] -= s_small * b[j];
    t[4] = now() - t[4];

    if (it == 0) continue;
    for (int k = 0; k < NKERN; k++) {
      sum[k] += t[k];
      if (t[k] < best[k]) best[k] = t[k];
    }
  }

  printf("# threads=%d array_size=%ld elementi (%.1f MB per array) NTIMES=%d\n",
         threads, n, n * sizeof(double) / 1e6, NTIMES);
  printf("%-10s %12s %12s %12s\n", "Kernel", "BestGB/s", "AvgGB/s", "BestTime[s]");
  for (int k = 0; k < NKERN; k++) {
    double gb = bytes[k] * n / 1e9;
    printf("%-10s %12.3f %12.3f %12.6f\n", name[k], gb / best[k],
           gb / (sum[k] / (NTIMES - 1)), best[k]);
  }
  /* Impedisce l'eliminazione del codice morto. */
  printf("# check %g\n", a[n / 2] + b[n / 3] + c[n / 5]);
  return 0;
}
