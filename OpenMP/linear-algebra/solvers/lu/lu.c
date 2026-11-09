/*
 * lu.c - Fattorizzazione LU in place (variante PolyBench: la riga k viene
 * divisa per il pivot), con le versioni di ottimizzazione OpenMP.
 *
 *   -DVERSION=0  seriale (baseline)
 *   -DVERSION=1  parallel for sui cicli interni
 *   -DVERSION=2  una sola regione parallela + omp for
 *   -DVERSION=3  + reciproco, hoisting, restrict, simd
 *   -DVERSION=4  + schedule(static,1) (riga i -> thread i % T, sempre)
 *
 * Uso: ./lu_vX [N=2000] [REPS=5] [CHECK=0]
 * Output (CSV): version,n,threads,min_s,median_s,gflops,maxreldiff,residual
 */
#define _POSIX_C_SOURCE 200112L
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef VERSION
#define VERSION 0
#endif

/* Accesso a A[i][j] con leading dimension ld (variabili `a` e `ld` in scope). */
#define A(i, j) a[(size_t)(i) * ld + (j)]

/* ---------- Inizializzazione deterministica, diagonalmente dominante ---------- */
/* Niente pivoting nel kernel: la dominanza diagonale evita pivot nulli. */
static inline double init_val(size_t i, size_t j, size_t n) {
    uint64_t x = (uint64_t)i * 0x9E3779B97F4A7C15ULL ^ (uint64_t)(j + 1) * 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 31;
    x *= 0x94D049BB133111EBULL;
    x ^= x >> 29;
    double r = (double)(x >> 11) * (1.0 / 9007199254740992.0); /* [0,1) */
    return (i == j) ? (double)n + r : r;
}

static void init(int n, size_t ld, double *a) {
    for (int i = 0; i < n; i++)
        for (size_t j = 0; j < ld; j++)
            A(i, j) = (j < (size_t)n) ? init_val(i, j, n) : 0.0;
}

static double *alloc_matrix(size_t n, size_t ld) {
    size_t bytes = n * ld * sizeof(double);
    bytes = (bytes + 63) / 64 * 64;
    void *p = NULL;
    if (posix_memalign(&p, 64, bytes) != 0) {
        fprintf(stderr, "allocazione fallita\n");
        exit(1);
    }
    return (double *)p;
}

/* ---------- Kernel seriale (sempre presente: riferimento per --check) ---------- */
static void kernel_seq(int n, size_t ld, double *a) {
    int i, j, k;
    for (k = 0; k < n; k++) {
        for (j = k + 1; j < n; j++)
            A(k, j) = A(k, j) / A(k, k);
        for (i = k + 1; i < n; i++)
            for (j = k + 1; j < n; j++)
                A(i, j) = A(i, j) - A(i, k) * A(k, j);
    }
}

/* ---------- Kernel della versione scelta ---------- */
#if VERSION == 0

static void kernel(int n, size_t ld, double *a) { kernel_seq(n, ld, a); }

#elif VERSION == 1

static void kernel(int n, size_t ld, double *a) {
    int i, j, k;
    for (k = 0; k < n; k++) {
        #pragma omp parallel for
        for (j = k + 1; j < n; j++)
            A(k, j) = A(k, j) / A(k, k);

        #pragma omp parallel for private(j)
        for (i = k + 1; i < n; i++)
            for (j = k + 1; j < n; j++)
                A(i, j) = A(i, j) - A(i, k) * A(k, j);
    }
}

#elif VERSION == 2

static void kernel(int n, size_t ld, double *a) {
    int i, j, k;
    #pragma omp parallel private(k, i, j)
    for (k = 0; k < n; k++) {
        #pragma omp for
        for (j = k + 1; j < n; j++)
            A(k, j) = A(k, j) / A(k, k);
        /* barriera implicita: la riga k deve essere normalizzata */

        #pragma omp for
        for (i = k + 1; i < n; i++)
            for (j = k + 1; j < n; j++)
                A(i, j) = A(i, j) - A(i, k) * A(k, j);
        /* barriera implicita: il passo k+1 usa la riga k+1 aggiornata */
    }
}

#else /* VERSION 3, 4 */

static void kernel(int n, size_t ld, double *a) {
    #pragma omp parallel
    for (int k = 0; k < n; k++) {
        const double inv = 1.0 / A(k, k); /* (a) una divisione, poi moltiplicazioni */

        #pragma omp for simd schedule(static)
        for (int j = k + 1; j < n; j++)
            A(k, j) *= inv;

#if VERSION == 4
        #pragma omp for schedule(static, 1) /* (b) riga i sempre allo stesso thread */
#else
        #pragma omp for schedule(static)
#endif
        for (int i = k + 1; i < n; i++) {
            const double aik = A(i, k);              /* (c) hoisting */
            double *restrict Ai = &A(i, 0);          /* (d) no aliasing */
            const double *restrict Ak = &A(k, 0);
            #pragma omp simd                         /* (e) vettorizzazione */
            for (int j = k + 1; j < n; j++)
                Ai[j] -= aik * Ak[j];
        }
    }
}

#endif

/* ---------- Verifica: residuo max|L*U - A0| / max|A0| ---------- */
/* L = parte triangolare inferiore (pivot inclusi), U = triangolare superiore
 * con diagonale unitaria implicita. */
static double residual(int n, size_t ld, const double *a) {
    double maxerr = 0.0, maxabs = 0.0;
    #pragma omp parallel for reduction(max : maxerr, maxabs) schedule(static)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int m = i < j ? i : j;
            double s = 0.0;
            for (int k = 0; k <= m; k++)
                s += A(i, k) * (k < j ? A(k, j) : 1.0);
            double orig = init_val(i, j, n);
            double e = fabs(s - orig);
            if (e > maxerr) maxerr = e;
            if (fabs(orig) > maxabs) maxabs = fabs(orig);
        }
    }
    return maxerr / maxabs;
}

static int cmp_double(const void *x, const void *y) {
    double a = *(const double *)x, b = *(const double *)y;
    return (a > b) - (a < b);
}

int main(int argc, char **argv) {
    int n = argc > 1 ? atoi(argv[1]) : 2000;
    int reps = argc > 2 ? atoi(argv[2]) : 5;
    int check = argc > 3 ? atoi(argv[3]) : 0;
    if (n < 2 || reps < 1) {
        fprintf(stderr, "uso: %s [N>=2] [REPS>=1] [CHECK]\n", argv[0]);
        return 1;
    }

    size_t ld = (size_t)n;

    double *a = alloc_matrix(n, ld);
    double *t = malloc((size_t)reps * sizeof(double));

    init(n, ld, a);       /* warm-up (page fault, frequenze), non misurato */
    kernel(n, ld, a);

    for (int r = 0; r < reps; r++) {
        init(n, ld, a);
        double t0 = omp_get_wtime();
        kernel(n, ld, a);
        t[r] = omp_get_wtime() - t0;
    }

    double err = 0.0, res = 0.0;
    if (check) {
        res = residual(n, ld, a);
        double *b = alloc_matrix(n, n);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                b[(size_t)i * n + j] = init_val(i, j, n);
        kernel_seq(n, (size_t)n, b);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) {
                double x = a[(size_t)i * ld + j], y = b[(size_t)i * n + j];
                double d = fabs(x - y) / (fabs(y) + 1e-30);
                if (d > err) err = d;
            }
        free(b);
    }

    qsort(t, (size_t)reps, sizeof(double), cmp_double);
    double tmin = t[0], tmed = t[reps / 2];
    double gflops = (2.0 / 3.0) * (double)n * n * n / tmin / 1e9;
#if VERSION == 0
    int threads = 1;
#else
    int threads = omp_get_max_threads();
#endif

    if (check)
        printf("%d,%d,%d,%.6f,%.6f,%.3f,%.3e,%.3e\n", VERSION, n, threads, tmin, tmed, gflops, err, res);
    else
        printf("%d,%d,%d,%.6f,%.6f,%.3f,NA,NA\n", VERSION, n, threads, tmin, tmed, gflops);

    free(t);
    free(a);
    return 0;
}