# Fattorizzazione LU con OpenMP: versioni ottimizzate

Kernel: fattorizzazione LU in place, senza pivoting, con la riga `k` divisa per il pivot (variante PolyBench). Le versioni descritte sono cumulative: ogni versione parte dalla precedente e aggiunge o cambia una tecnica. Le sorgenti disponibili sono `lu.c` (versione 0), `lu_v3.c` e `lu_v4.c`; le versioni 1 e 2 sono qui descritte come passaggi intermedi, ma non hanno una sorgente separata.

Nei frammenti `A(i,j)` indica `a[i*ld + j]`, dove `ld` è la leading dimension della matrice.

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

**Cosa fa e cosa aspettarsi:** è il codice originale, compilato con `-O3 -march=native`. Non ha overhead OpenMP ed è il riferimento per correttezza, speedup ed efficienza. Le divisioni per ogni elemento della riga e il lavoro seriale sull'aggiornamento della matrice sono i principali costi che le versioni successive cercano di ridurre.

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

**Cosa cambia e cosa aspettarsi:** `k` resta sequenziale perché ogni passo dipende dai precedenti. Si distribuiscono invece tra i thread la normalizzazione della riga e l'aggiornamento delle righe sottostanti: ogni `i` scrive una riga distinta. `private(j)` rende privato l'indice del ciclo interno. Si introduce parallelismo, ma si crea e si termina un team di thread due volte per ogni `k`; per matrici piccole tale overhead può superare il lavoro utile.

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
    /* barriera implicita: il passo k+1 usa la riga aggiornata */
}
```

**Cosa cambia e cosa aspettarsi:** il team viene creato una sola volta e riutilizzato per tutti i passi `k`; `omp for` ridistribuisce il lavoro senza fork/join ripetuti. Le barriere implicite dopo i due cicli sono necessarie: la riga deve essere normalizzata prima dell'aggiornamento, e ogni aggiornamento deve terminare prima del passo successivo. Ci si aspetta un guadagno soprattutto quando la matrice è abbastanza grande da rendere rilevante l'overhead ripetuto della Versione 1. Per matrici piccole l'overhead di avvio del singolo team rimane comunque.

## Versione 3: riduzione del costo per elemento e SIMD

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

**Cosa cambia rispetto alla Versione 2:** mantiene un solo team e le stesse dipendenze/barriere, ma ottimizza le operazioni nei cicli:

- **Reciproco del pivot:** calcola una volta `1 / A(k,k)` e moltiplica gli elementi della riga per quel valore, invece di eseguire una divisione per elemento. Ci si aspetta meno costo aritmetico, soprattutto dove la divisione è più lenta della moltiplicazione. L'arrotondamento può differire dalla divisione ripetuta di circa un ulp.
- **Hoisting di `A(i,k)`:** il valore, costante rispetto a `j`, viene caricato una sola volta per riga invece che riletto a ogni iterazione.
- **`restrict`:** dichiara che i puntatori alle righe `i` e `k` non si sovrappongono; ciò aiuta il compilatore a dimostrare l'assenza di aliasing e a vettorizzare l'aggiornamento.
- **`omp simd`:** abilita l'esecuzione vettoriale sul ciclo `j`, che percorre elementi contigui della riga. `omp for simd` combina la distribuzione della riga normalizzata tra thread con la SIMD.
- **`schedule(static)`:** assegna a ciascun thread un blocco contiguo di iterazioni, con basso overhead di scheduling e accessi per thread relativamente contigui.

Insieme, queste modifiche mirano a diminuire il costo dell'aritmetica e a sfruttare i vettori SIMD del processore; il guadagno effettivo dipende dal compilatore, dalla CPU e dalla dimensione della matrice.

## Versione 4: scheduling ciclico `static,1`

La Versione 4 mantiene le ottimizzazioni aritmetiche, `restrict`, SIMD e la singola regione parallela della Versione 3. Cambia soltanto la distribuzione delle righe del ciclo di aggiornamento:

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

**Differenza rispetto alla Versione 3:** `schedule(static)` senza chunk esplicito distribuisce normalmente blocchi contigui di righe ai thread; `schedule(static,1)` assegna invece una riga per volta in modo ciclico tra i thread. La proprietà del lavoro è quindi diversa: nella Versione 3 ogni thread elabora un blocco, nella Versione 4 le righe vengono intercalate.

**Prestazioni attese e compromessi:** il ciclo per righe di questo kernel ha un costo simile per ogni `i`, quindi non ci si aspetta automaticamente un migliore bilanciamento con `static,1`. La distribuzione ciclica può aiutare in varianti con costi delle righe non uniformi o in alcune configurazioni di memoria; d'altra parte può peggiorare la località rispetto ai blocchi contigui e aggiunge un scheduling più granulare. Inoltre, poiché l'inizio dell'intervallo (`k+1`) cambia a ogni passo, `static,1` non garantisce che una data riga resti assegnata allo stesso thread nei passi successivi. La Versione 4 è dunque un'alternativa da misurare, non un miglioramento garantito.

## Confronto sintetico

| Versione | Modifica principale | Miglioramento atteso | Possibile costo/limite |
|---|---|---|---|
| 0 | Kernel seriale | Riferimento | Nessun parallelismo |
| 1 | `parallel for` interni | Sfrutta più thread | Fork/join ripetuti |
| 2 | Una regione parallela | Riduce l'overhead di fork/join | Team OpenMP anche per lavoro piccolo |
| 3 | Reciproco, hoisting, `restrict`, SIMD e `static` a blocchi | Meno divisioni e più lavoro vettoriale, con scheduling a basso overhead | Arrotondamento diverso; efficacia dipendente dal compilatore e dall'hardware |
| 4 | `static,1` sulle righe aggiornate | Potenziale bilanciamento quando il costo per riga varia | Può ridurre la località; nel kernel uniforme può non portare benefici |

## Valutazione delle prestazioni

### Prerequisiti e preparazione

Usare una macchina Linux con `perf`, Valgrind (strumento Cachegrind) e un compilatore C con supporto OpenMP. `perf` deve poter accedere ai contatori hardware: disponibilità degli eventi e permessi dipendono dalla macchina e dalla sua configurazione. macOS non fornisce normalmente il `perf` di Linux; per queste misure usare un ambiente Linux compatibile con entrambi gli strumenti. La presenza dei comandi non garantisce che tutti gli eventi siano supportati o accessibili.

Lo script [`perf_valgrind.sh`](./perf_valgrind.sh) compila direttamente le tre sorgenti disponibili, insieme a `polybench.c`; non misura le versioni 1 e 2. Per le compilazioni segue i valori predefiniti del Makefile (`gcc`, `-O2 -g -fopenmp` e `-lm`) e prova per default prima 4 e poi 8 thread. Eseguire i confronti sulla stessa macchina, con le stesse opzioni e lo stesso `N`. Se necessario, si possono impostare `CC`, `CFLAGS` e `OMPFLAGS` nell'ambiente prima del comando.

**Correttezza prima delle prestazioni:** l'inizializzazione corrente produce una matrice di rango uno, quindi singolare per `N > 1`. La LU senza pivoting può incontrare pivot nulli o numericamente molto piccoli, producendo valori non finiti o risultati instabili. Lo script non verifica la correttezza né il residuo: confrontare le prestazioni non dimostra che la fattorizzazione sia valida. Per una valutazione attendibile del kernel, verificare separatamente il risultato rispetto a un riferimento e il residuo su matrici con pivot non nulli, usando lo stesso input per ogni versione; il solo confronto dell'output stampato a due decimali non basta.

### Esecuzione

Dalla cartella del solver:

```bash
./perf_valgrind.sh [N] [PERF_REPS] ["THREADS"] [OUT.csv] [VALGRIND_REPS]
```

I parametri sono posizionali e hanno questi valori predefiniti:

| Parametro | Default | Significato |
|---|---:|---|
| `N` | `2000` | Dimensione della matrice quadrata, fissata in compilazione |
| `PERF_REPS` | `5` | Ripetizioni di `perf` per versione e numero di thread |
| `THREADS` | `"4 8"` | Elenco ordinato dei valori di `OMP_NUM_THREADS`, tra virgolette |
| `OUT.csv` | `lu_results.csv` | File CSV; se esiste, viene sovrascritto |
| `VALGRIND_REPS` | `1` | Ripetizioni di Cachegrind per versione e numero di thread |

Per una prima prova, su una matrice piccola:

```bash
./perf_valgrind.sh 128 3 "4 8" lu_prova.csv 1
```

Per una misura più rappresentativa, ad esempio:

```bash
OMP_DYNAMIC=FALSE OMP_PROC_BIND=TRUE OMP_PLACES=cores \
  ./perf_valgrind.sh 2000 5 "4 8" lu_results.csv 1
```

La seconda esecuzione può richiedere molto tempo: Cachegrind rallenta notevolmente il programma. Scegliere `N` e i thread in funzione delle risorse disponibili, iniziando con pochi thread e un input piccolo. L'impostazione di `OMP_DYNAMIC=FALSE` e dell'affinità aiuta a rendere i test ripetibili; annotare CPU, compilatore, opzioni, sistema, numero di core e variabili OpenMP usate. Limitare gli altri carichi sulla macchina e ripetere l'esperimento in condizioni analoghe.

### Lettura del CSV e confronto

Ogni riga rappresenta una combinazione di `tool`, `version`, `n`, `threads` e `rep`. `v0` è seriale: viene comunque eseguita per ogni valore di `threads`, ma non usa più thread per il kernel. Controllare `exit_status`: usare solo esecuzioni riuscite (`0`) e considerare le celle vuote come dati mancanti, non come zeri. Messaggi di avviso, contatori assenti o errori di permesso rendono la relativa misura inutilizzabile; verificare la disponibilità degli eventi prima di trarre conclusioni.

- Le righe `perf` riportano il tempo trascorso in `wall_seconds`, `task_clock_ms`, cicli, istruzioni, riferimenti e miss di cache e branch miss, quando disponibili. `wall_seconds` misura l'intera invocazione di `perf` e del programma, incluso l'avvio e l'inizializzazione: **non è il tempo del solo kernel**. L'overhead può incidere soprattutto per input piccoli. `task_clock_ms` è tempo CPU aggregato, non tempo trascorso, e può aumentare usando più thread.
- Le righe `valgrind` riportano i contatori Cachegrind: `vg_ir` istruzioni, `vg_dr`/`vg_dw` letture/scritture dati, e i rispettivi miss simulati `i1`, `d1` e `il`/`dl` (ultimo livello). Sono stime della cache simulata, non misure dei contatori hardware. Valgrind può alterare notevolmente tempi e scheduling dei thread: non confrontare il suo `wall_seconds` con il tempo nativo né usarlo per calcolare speedup paralleli.
- I contatori riguardano l'intero processo, non soltanto il kernel. Su più thread riflettono anche il runtime OpenMP; eventi diversi possono essere indisponibili o avere significati dipendenti dalla CPU. Un numero inferiore di miss o istruzioni non implica necessariamente un tempo inferiore.

Per ogni `N` e versione, confrontare la **mediana** delle ripetizioni riuscite di `perf` e riportare anche la variabilità. A parità di condizioni, calcolare lo speedup della versione parallela con `T` thread come `mediana(wall_seconds di v0 con 1 thread) / mediana(wall_seconds della versione con T thread)`; l'efficienza è `speedup / T`. Sono speedup del processo misurato, comprensivi dell'overhead indicato sopra, non del solo kernel. Confrontare inoltre v3 e v4 allo stesso `N` e numero di thread per valutare l'effetto di `static,1`. Differenze inferiori alla variabilità delle misure non dimostrano un miglioramento; lo script salva i dati grezzi e non calcola automaticamente mediana, speedup o efficienza.
