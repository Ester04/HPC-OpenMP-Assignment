# Fattorizzazione LU con OpenMP: versioni ottimizzate

Kernel: fattorizzazione LU in place, senza pivoting, con la riga `k` divisa per il pivot (variante PolyBench). Le versioni descritte sono cumulative: ogni versione parte dalla precedente e aggiunge o cambia una tecnica. Le sorgenti disponibili sono `lu.c` (versione 0), `lu_v2.c`, `lu_v2_steps.c`, `lu_v3.c` e `lu_v4.c`; la versione 1 resta un passaggio intermedio senza sorgente separata.

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

**Cosa fa e cosa aspettarsi:** è il codice originale. Non ha overhead OpenMP ed è il riferimento per correttezza, speedup ed efficienza. Le divisioni per ogni elemento della riga e il lavoro seriale sull'aggiornamento della matrice sono i principali costi che le versioni successive cercano di ridurre.

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

## Versione 2: una sola regione parallela (`lu_v2.c`)

Questa versione è implementata in `lu_v2.c` e compilata dal benchmark, così da misurare il passaggio dalla baseline seriale alla prima versione parallela con un solo team.

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

### Passaggi intermedi da v2 a v3

Il sorgente [`lu_v2_steps.c`](./lu_v2_steps.c) è compilato cinque volte con `LU_STEP` da 1 a 5. Ogni binario aggiunge una modifica alla variante precedente, mantenendo lo stesso input e il team parallelo di v2:

| Nome nel CSV | `LU_STEP` | Modifica aggiunta rispetto alla versione precedente |
|---|---:|---|
| `v2_simd` | 1 | `omp for simd` sulla normalizzazione e `omp simd` sull'aggiornamento, mantenendo le divisioni. |
| `v2_recip` | 2 | Reciproco del pivot calcolato da ciascun thread e moltiplicazione al posto delle divisioni. |
| `v2_hoist` | 3 | Caricamento di `A[i][k]` una volta per riga. |
| `v2_rows` | 4 | Puntatori locali `Ai` e `Ak` alle righe, senza `restrict`. |
| `v2_restrict` | 5 | Aggiunta di `restrict` ai puntatori delle righe. |
| `v3` | — | Come passo finale, scheduling `static` esplicito nei due cicli; v3 usa inoltre indici locali alla regione invece di indici dichiarati `private`. |

Confrontare ogni coppia adiacente allo stesso `N` e numero di thread, usando la mediana delle ripetizioni e la variabilità. La sequenza misura l'effetto **incrementale nel contesto delle modifiche precedenti**, non il contributo universale di ogni tecnica: il compilatore potrebbe già applicare trasformazioni equivalenti, e modifiche diverse possono interagire. `omp simd` non garantisce da solo che il compilatore generi istruzioni vettoriali: verificare con il suo report di vettorizzazione. Lo scheduling predefinito senza clausola è dipendente dall'implementazione e potrebbe già coincidere con `static`.

Per una verifica preliminare delle cinque varianti su un input non singolare, sulla macchina Linux con GCC e OpenMP, dalla cartella LU:

```bash
for step in 1 2 3 4 5; do
  if ! gcc -O2 -g -fopenmp -DN=16 -DLU_STEP="$step" -I../../../utilities \
    test_lu_v2_steps.c ../../../utilities/polybench.c -lm -o "lu_step_test_$step"; then
    echo "Compilazione fallita per LU_STEP=$step" >&2
    break
  fi
  for threads in 1 4 8; do
    if ! OMP_DYNAMIC=FALSE OMP_NUM_THREADS="$threads" "./lu_step_test_$step"; then
      echo "Test fallito per LU_STEP=$step, thread=$threads" >&2
      break 2
    fi
  done
done
```

Il test confronta gli elementi con un riferimento seriale su matrici diagonalmente dominanti di dimensione 1 e 16, con tolleranza numerica, e segnala eventuali errori. Non sostituisce la verifica dei dataset effettivi né modifica l'input singolare usato dal benchmark.

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

Lo script [`perf_valgrind.sh`](./perf_valgrind.sh) compila nove binari per dataset: `v0`, `v2`, i cinque passaggi intermedi, `v3` e `v4`, insieme a `polybench.c`; la versione 1 non ha una sorgente separata. Per default usa tutti e cinque i dataset di [`lu.h`](./lu.h): MINI (32), SMALL (128), STANDARD (1024), LARGE (2000), EXTRALARGE (4000). Le dimensioni sono ricavate dal preprocessore del compilatore, così restano coerenti con il file `.h`. Per le compilazioni usa `gcc`, `-O2 -g -fopenmp` e `-lm`, con il timer PolyBench abilitato. Richiede per default **4 thread e 3 ripetizioni** per ciascuno strumento. Eseguire i confronti sulla stessa macchina, con le stesse opzioni e lo stesso `N`. Se necessario, si possono impostare `CC`, `CFLAGS` e `OMPFLAGS` nell'ambiente prima del comando.

**Correttezza prima delle prestazioni:** l'inizializzazione corrente produce una matrice di rango uno, quindi singolare per `N > 1`. La LU senza pivoting può incontrare pivot nulli o numericamente molto piccoli, producendo valori non finiti o risultati instabili. Lo script non verifica la correttezza né il residuo: confrontare le prestazioni non dimostra che la fattorizzazione sia valida. Per una valutazione attendibile del kernel, verificare separatamente il risultato rispetto a un riferimento e il residuo su matrici con pivot non nulli, usando lo stesso input per ogni versione; il solo confronto dell'output stampato a due decimali non basta.

### Esecuzione

Dalla cartella del solver:

```bash
./perf_valgrind.sh ["DATASETS"] [PERF_REPS] ["THREADS"] [OUT.csv] [VALGRIND_REPS]
```

I parametri sono posizionali e hanno questi valori predefiniti:

| Parametro | Default | Significato |
|---|---:|---|
| `DATASETS` | `"MINI SMALL STANDARD LARGE EXTRALARGE"` | Elenco dei dataset di `lu.h` o dimensioni numeriche, tra virgolette |
| `PERF_REPS` | `3` | Ripetizioni di `perf` per versione e numero di thread |
| `THREADS` | `"4"` | Elenco dei valori richiesti di `OMP_NUM_THREADS`, tra virgolette |
| `OUT.csv` | `lu_kernel_results.csv` | Nuovo CSV; un file esistente non viene sovrascritto |
| `VALGRIND_REPS` | `3` | Ripetizioni Cachegrind; `0` le disabilita |

Per una prima prova, su una matrice piccola (è ancora possibile specificare direttamente `128` come primo argomento):

```bash
./perf_valgrind.sh MINI 3 4 lu_prova_kernel.csv 0
```

Per misurare tutti e cinque i dataset nello stesso CSV:

```bash
OMP_DYNAMIC=FALSE OMP_PROC_BIND=TRUE OMP_PLACES=cores \
  ./perf_valgrind.sh
```

Con i default sono previste **135 esecuzioni `perf` e 135 Cachegrind**; Cachegrind può richiedere moltissimo tempo, soprattutto per EXTRALARGE. Per misurare soltanto con `perf`, senza richiedere Valgrind:

```bash
OMP_PROC_BIND=TRUE OMP_PLACES=cores \
  ./perf_valgrind.sh "" 3 4 lu_kernel_perf.csv 0
```

Un sottoinsieme si può selezionare, per esempio, con `"MINI SMALL STANDARD"` e `LU_VERSIONS="v0 v2 v2_simd v3 v4"`. Lo script imposta `OMP_DYNAMIC=FALSE`; per ripetibilità scegliere anche un'affinità adatta alla macchina e annotare CPU, compilatore, flag, sistema, core e variabili OpenMP. Il valore 4 è richiesto, non verificato come dimensione effettiva del team.

`LU_APPEND=1` consente di aggiungere combinazioni assenti a un CSV con il **nuovo** schema. I vecchi CSV non contengono `kernel_seconds` e vengono rifiutati, per non mescolare misure incompatibili. L'append non verifica macchina, flag o condizioni. Se una prova fallisce lo script si ferma segnalando l'errore e il file può restare parziale; non riprende automaticamente combinazioni già presenti.

### Lettura del CSV e confronto

Ogni riga rappresenta una combinazione di `tool`, `version`, `n`, `threads` e `rep`. `v0` è seriale: viene comunque eseguita per ogni valore di `threads`, ma non usa più thread per il kernel. Controllare `exit_status`: usare solo esecuzioni riuscite (`0`) e considerare le celle vuote come dati mancanti, non come zeri. Messaggi di avviso, contatori assenti o errori di permesso rendono la relativa misura inutilizzabile; verificare la disponibilità degli eventi prima di trarre conclusioni.

- `kernel_seconds` è il tempo trascorso tra `polybench_start_instruments` e `polybench_stop_instruments`, attorno alla chiamata `kernel_lu`: esclude inizializzazione, stampa e deallocazione, ma comprende chiamata, creazione del team OpenMP e sincronizzazioni. PolyBench usa `gettimeofday` e stampa sei decimali; non è un clock monotono e la precisione effettiva, soprattutto per dataset piccoli, è limitata. Lo script disabilita il flush cache PolyBench prima del timer, per evitare che crei un team OpenMP prima del kernel.
- Le righe `perf` riportano anche `wall_seconds`, l'intera invocazione di `perf` e del programma, oltre a `task_clock_ms`, cicli, istruzioni, riferimenti/miss di cache e branch miss, quando disponibili. Anche `kernel_seconds` è rilevato **durante** `perf` e può risentirne. `task_clock_ms` è tempo CPU aggregato, non tempo trascorso.
- Le righe `valgrind` riportano i contatori Cachegrind: `vg_ir` istruzioni, `vg_dr`/`vg_dw` letture/scritture dati e miss simulati. Valgrind altera tempi e scheduling: **non usare né `kernel_seconds` né `wall_seconds` di queste righe come tempi nativi o per speedup**.
- I contatori riguardano l'intero processo, non soltanto il kernel. Su più thread riflettono anche il runtime OpenMP; eventi diversi possono essere indisponibili o avere significati dipendenti dalla CPU. Un numero inferiore di miss o istruzioni non implica necessariamente un tempo inferiore.

Per ogni `N` e versione, confrontare la **mediana di `kernel_seconds` delle tre righe `perf`** e riportare anche min–max. Calcolare lo speedup come `mediana(kernel_seconds di v0) / mediana(kernel_seconds della versione)` con lo stesso `N` e le stesse condizioni. `v0` resta seriale. Tre prove non bastano per conclusioni statistiche robuste; i contatori di processo non isolano il kernel.

### Grafici

Il generatore esistente [`plot_lu_results.py`](./plot_lu_results.py) usa **`wall_seconds`, non `kernel_seconds`**. Anche se accetta il nuovo CSV, i suoi grafici temporali e speedup restano relativi all'intero processo: non presentarli come misure del kernel.

```bash
python3 plot_lu_results.py lu_kernel_results.csv
# Opzionale: scegliere la cartella di destinazione
python3 plot_lu_results.py lu_kernel_results.csv --output-dir grafici
```

Le ripetizioni riuscite vengono aggregate con la mediana. Un grafico Cachegrind descrive istruzioni simulate e non va interpretato come misura dei tempi nativi.
