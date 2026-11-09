# Fattorizzazione LU con OpenMP: versioni ottimizzate

Kernel: fattorizzazione LU in place, senza pivoting, con la riga `k` divisa per il pivot (variante PolyBench). Ogni versione parte dalla precedente. Nel codice si sceglie con `-DVERSION=N`.

Nei frammenti `A(i,j)` indica `a[i*ld + j]`, dove `ld` è la leading dimension della matrice.

---

## Versione 0: baseline seriale

```c
for (k = 0; k < n; k++) {
    for (j = k + 1; j < n; j++)
        A(k, j) = A(k, j) / A(k, k);
    for (i = k + 1; i < n; i++)
        for (j = k + 1; j < n; j++)
            A(i, j) = A(i, j) - A(i, k) * A(k, j);
}
```

**Spiegazione:** codice originale, compilato con `-O3 -march=native`. È il riferimento per speedup, efficienza e correttezza di tutte le altre versioni.

---

## Versione 1: parallelizzazione dei cicli interni

```c
for (k = 0; k < n; k++) {
    #pragma omp parallel for
    for (j = k + 1; j < n; j++)
        A(k, j) = A(k, j) / A(k, k);

    #pragma omp parallel for private(j)
    for (i = k + 1; i < n; i++)
        for (j = k + 1; j < n; j++)
            A(i, j) = A(i, j) - A(i, k) * A(k, j);
}
```

**Spiegazione:** il ciclo `k` resta seriale perché il passo `k+1` dipende dal passo `k`. Le iterazioni dei cicli interni sono indipendenti: ogni `i` scrive solo la propria riga e legge la riga `k` e la colonna `k`, che in quella fase non vengono modificate. `private(j)` è necessario, altrimenti l'indice interno è condiviso e causa una data race. Limite: il team di thread viene creato e distrutto `2N` volte.

---

## Versione 2: una sola regione parallela

```c
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
```

**Spiegazione:** il team di thread viene creato una sola volta. Ogni thread esegue il ciclo `k` con la propria copia privata, mentre `omp for` ripartisce solo il lavoro. Si elimina il costo di fork/join. Le due barriere implicite sono dipendenze vere, quindi non si può usare `nowait`.

---

## Versione 3: micro-ottimizzazioni del ciclo

```c
#pragma omp parallel
for (int k = 0; k < n; k++) {
    const double inv = 1.0 / A(k, k);

    #pragma omp for simd schedule(static)
    for (int j = k + 1; j < n; j++)
        A(k, j) *= inv;

    #pragma omp for schedule(static)
    for (int i = k + 1; i < n; i++) {
        const double aik = A(i, k);
        double *restrict Ai = &A(i, 0);
        const double *restrict Ak = &A(k, 0);
        #pragma omp simd
        for (int j = k + 1; j < n; j++)
            Ai[j] -= aik * Ak[j];
    }
}
```

**Spiegazione:**
- **Reciproco:** una divisione e `N-k` moltiplicazioni al posto di `N-k` divisioni. Il risultato può cambiare di circa 1 ulp.
- **Hoisting di `A(i,k)`:** il valore non dipende da `j`, quindi resta in un registro.
- **`restrict`:** garantisce al compilatore che le righe `i` e `k` non si sovrappongono, sbloccando vettorizzazione e FMA.
- **`omp simd`:** forza la vettorizzazione del ciclo interno, che ha accessi contigui.

---

## Versione 4: scheduling ciclico `static,1`

```c
#pragma omp for schedule(static, 1)
for (int i = k + 1; i < n; i++) {
    const double aik = A(i, k);
    double *restrict Ai = &A(i, 0);
    const double *restrict Ak = &A(k, 0);
    #pragma omp simd
    for (int j = k + 1; j < n; j++)
        Ai[j] -= aik * Ak[j];
}
```

**Spiegazione:** con `static` a blocchi il range di `i` si restringe a ogni `k`, quindi una riga cambia thread da un passo all'altro e perde la cache. Con `static,1` la riga `i` va sempre al thread `i % T`: ogni thread riusa le proprie righe in L1/L2 e il bilanciamento resta buono anche con range piccolo. `dynamic` e `guided` perderebbero questa affinità e avrebbero overhead di coda, mentre il costo per riga è già uniforme.

---

## Riepilogo

| Versione | Modifica | Obiettivo |
|---|---|---|
| 0 | Seriale | Baseline |
| 1 | `parallel for` sui cicli interni | Parallelismo base |
| 2 | Regione parallela unica | Eliminare fork/join |
| 3 | Reciproco, hoisting, `restrict`, `simd` | Meno costo per iterazione, SIMD |
| 4 | `schedule(static,1)` | Affinità riga→thread, cache |

Conviene misurare ogni versione separatamente (`make bench`): una modifica che non migliora oltre la variabilità delle misure va tolta.