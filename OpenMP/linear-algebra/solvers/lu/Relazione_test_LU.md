# Valutazione delle ottimizzazioni della fattorizzazione LU

## Sintesi

Questa relazione analizza [`lu_results_intermedi.csv`](./lu_results_intermedi.csv)
seguendo la progressione dalla **versione originale seriale** (`v0`) alla
versione parallela con **pragma OpenMP** (`v2`), alla variante con direttive
**SIMD** (`v2_simd`), fino alle micro-ottimizzazioni cumulative e alle due
politiche di scheduling. Nei grafici della presentazione **static** indica
`v3` e **static,1** indica `v4`; questi rimangono i nomi nel CSV.

A `N=4000`, nel gruppo con 4 thread richiesti, le mediane del tempo di processo
sono 60.509 s per l'originale, 39.301 s per OpenMP e 54.055 s per SIMD. Il
rapporto originale/OpenMP è 1.54×; il confronto OpenMP/SIMD attraversa però
campagne diverse. `static` e `static,1` risultano rispettivamente a 38.566 s
e 38.536 s. **Non si può attribuire il divario tra SIMD e static alle sole
modifiche del codice**, né usare questi dati come prova di correttezza.

Il CSV contiene 450 misure `perf` (cinque ripetizioni per ogni versione,
dimensione e numero di thread richiesto) e 71 misure Cachegrind. Tutte le
combinazioni `perf` previste sono presenti; i contatori Cachegrind sono
incompleti. Il CSV non registra le informazioni necessarie per verificare che
le campagne siano state eseguite sulla stessa macchina, con gli stessi flag e
nelle stesse condizioni. Perciò i confronti che attraversano le campagne sono
indicativi, non conclusivi.

> **Limite fondamentale:** l'inizializzazione corrente crea una matrice di
> rango uno, singolare per `N > 1`; il kernel non usa pivoting. I risultati
> misurano quindi il comportamento del programma su questo input, non
> dimostrano prestazioni di una fattorizzazione LU numericamente corretta.

## 1. Versioni e ottimizzazioni

La fattorizzazione procede per pivot `k`. A ogni passo si normalizza la riga
del pivot e poi si aggiornano le righe sottostanti. Il ciclo su `k` è
dipendente dai passi precedenti e rimane sequenziale; il parallelismo è nei
cicli sulle colonne della riga e sulle righe da aggiornare.

Per `j > k` e `i > k`, i due passi del kernel sono:

```text
A[k][j] <- A[k][j] / A[k][k]          normalizzazione della riga pivot
A[i][j] <- A[i][j] - A[i][k] A[k][j]  aggiornamento delle altre righe
```

Le iterazioni di aggiornamento su righe diverse scrivono aree distinte, ma
usano tutte la riga pivot. La sua normalizzazione deve quindi terminare prima
dell'aggiornamento; anche l'intero aggiornamento deve terminare prima del
pivot successivo. Il lavoro complessivo cresce come **O(N³)**, mentre
l'inizializzazione della matrice è **O(N²)**. All'avanzare di `k` i cicli si
accorciano: sincronizzazione e lavoro residuo limitano la scalabilità.

| Versione / passaggio | Modifica | Motivazione e limite |
|---|---|---|
| v0 | Kernel seriale in [`lu.c`](./lu.c). | Baseline: nessun costo OpenMP, ma nessun parallelismo. |
| v1 | `parallel for` sui cicli interni. | Distribuisce il lavoro indipendente, ma creare e terminare team a ogni pivot può costare più del lavoro sulle matrici piccole. Non è presente una misura v1 separata. |
| v2 | Una regione `parallel` persistente con `omp for` in [`lu_v2.c`](./lu_v2.c). | Riutilizza il team per tutti i pivot. Le barriere implicite sono necessarie: la normalizzazione deve precedere l'aggiornamento e questo deve terminare prima del pivot successivo. |
| v2 → `v2_simd` | `omp for simd` sulla normalizzazione e `omp simd` sull'aggiornamento, mantenendo le divisioni. | Espone al compilatore iterazioni indipendenti e contigue in memoria. Una direttiva SIMD non garantisce da sola istruzioni vettoriali: serve controllare il report di vettorizzazione o il codice macchina. |
| `v2_simd` → `v2_recip` | Calcola il reciproco del pivot una volta per thread e moltiplica gli elementi della riga. | Riduce il numero di divisioni, operazione spesso più costosa della moltiplicazione. La forma reciproco × valore può arrotondare diversamente dalla divisione. |
| `v2_recip` → `v2_hoist` | Carica `A[i][k]` una volta per riga, fuori dal ciclo su `j`. | È un valore invariato nel ciclo interno: il riuso può evitare caricamenti ripetuti. Il compilatore potrebbe già effettuare questa trasformazione. |
| `v2_hoist` → `v2_rows` | Usa puntatori locali `Ai` e `Ak` alle righe. | Rende espliciti gli accessi lineari e può aiutare l'analisi degli indirizzi. Il guadagno dipende dal compilatore e dalla rappresentazione dell'array. |
| `v2_rows` → `v2_restrict` | Aggiunge `restrict` ai puntatori delle due righe. | Comunica al compilatore che gli accessi tramite i puntatori non si sovrappongono, favorendo l'ottimizzazione e la vettorizzazione. È una promessa del programma: se falsa, il comportamento non è definito. |
| `v2_restrict` → v3 | Aggiunge `schedule(static)` esplicito; gli indici sono locali alla regione. | La distribuzione statica assegna blocchi con basso overhead e buona località. Lo scheduling predefinito può già essere statico: il solo cambio di clausola potrebbe non introdurre una differenza effettiva. |
| v3 → v4 | Cambia lo scheduling del ciclo sulle righe in `schedule(static,1)`. | Assegna una riga per volta in modo ciclico. Può essere utile con carichi per-riga disuniformi, ma qui il lavoro delle righe è simile e la distribuzione a blocchi può avere località migliore. Non è un'ottimizzazione garantita. |

Le varianti intermedie sono cumulative: ogni passaggio aggiunge una modifica
alle precedenti. Il confronto fra due adiacenti stima l'effetto marginale nel
contesto di quelle già introdotte, non un effetto universale o isolato.
Implementazione dettagliata e protocollo sono descritti in
[`Versioni_ottimizzazione.md`](./Versioni_ottimizzazione.md); il sorgente delle
varianti intermedie è [`lu_v2_steps.c`](./lu_v2_steps.c).

### 1.1 Perché queste modifiche possono aiutare

1. **Originale seriale (`v0`).** Un solo flusso esegue tutto il lavoro, senza
   sincronizzazioni OpenMP; il compilatore può comunque applicare ottimizzazioni
   e vettorizzazione automatica. È il riferimento temporale, non una garanzia
   che il codice macchina sia privo di SIMD.
2. **Pragma OpenMP (`v2`).** Un'unica regione `parallel` crea il team e lo
   riusa a ogni pivot; i due `omp for` distribuiscono prima colonne e poi righe.
   Le barriere implicite preservano le dipendenze descritte sopra. Si evita
   così di creare ripetutamente il team, ma rimangono costo di distribuzione,
   attese e traffico di memoria. La `v1` con due `parallel for` per pivot è
   solo illustrativa: **non è presente come misura separata nel CSV**.
3. **SIMD (`v2_simd`).** `omp for simd` combina lavoro tra thread e corsie
   vettoriali nella normalizzazione; `omp simd` espone l'indipendenza delle
   iterazioni contigue su `j` nell'aggiornamento. Una singola istruzione
   vettoriale può elaborare più elementi; la direttiva non dimostra però
   da sola che siano state emesse istruzioni vettoriali. Occorrono il report
   del compilatore o l'analisi del codice macchina.
4. **Reciproco (`v2_recip`).** Ogni thread calcola `1 / A[k][k]` una volta per
   pivot e sostituisce le divisioni della riga con moltiplicazioni. È una
   riduzione del lavoro aritmetico attesa, non un guadagno garantito:
   compilatore e hardware contano, e l'arrotondamento può cambiare.
5. **Hoisting (`v2_hoist`).** `A[i][k]` non varia nel ciclo su `j` ed è
   caricato una volta per riga. Può eliminare caricamenti ripetuti, ma un
   compilatore potrebbe aver già applicato una trasformazione equivalente.
6. **Puntatori di riga (`v2_rows`) e `restrict` (`v2_restrict`).** `Ai` e `Ak`
   rendono esplicita la scansione contigua. `restrict` promette che gli
   accessi rilevanti non violino il contratto di aliasing, agevolando
   riordino e vettorizzazione; una promessa falsa causa comportamento non
   definito. In questo kernel `i > k` identifica righe distinte.
7. **Scheduling (`static`, `static,1`).** Senza chunk esplicito, `static`
   assegna blocchi di iterazioni approssimativamente uguali; `static,1`
   assegna singole righe ciclicamente. Il primo può favorire località e
   ridurre overhead; il secondo può distribuire meglio un carico disomogeneo.
   Qui, fissato `k`, le righe eseguono cicli di lunghezza simile, perciò il
   vantaggio di `static,1` non è scontato. Il passaggio da `v2_restrict` a
   `static` cambia anche la gestione degli indici, oltre alla clausola.

## 2. Come sono stati raccolti i risultati

Questa sezione descrive **la campagna storica presente nel CSV**, non la
configurazione attuale dello script. Lo script è stato successivamente
aggiornato per registrare separatamente `kernel_seconds`, con 4 thread
richiesti e 3 ripetizioni predefiniti; i risultati storici non contengono
questa misura e non vanno reinterpretati come tempi del solo kernel. La nuova
procedura è descritta in [Versioni_ottimizzazione.md](./Versioni_ottimizzazione.md).

Lo script [`perf_valgrind.sh`](./perf_valgrind.sh) compila le versioni e
misura i dataset definiti in [`lu.h`](./lu.h): MINI (`N=32`), SMALL (`128`),
STANDARD (`1024`), LARGE (`2000`) ed EXTRALARGE (`4000`). Le prove `perf`
usano 4 e 8 come valori di `OMP_NUM_THREADS`, con cinque ripetizioni. Questi
sono thread **richiesti**, non una prova che la macchina abbia altrettanti
core disponibili o che il runtime abbia creato il team previsto.

Lo script usa come default `gcc`, `-O2 -g`, `-fopenmp` e `-lm`, salvo override
delle variabili d'ambiente. Per ogni prova esegue `perf stat` con
`task-clock`, `cycles`, `instructions`, riferimenti/miss di cache e branch
misses. `wall_seconds` è ottenuto dal comando `time` attorno a `perf` e al
processo misurato: include avvio, inizializzazione, kernel, terminazione e
overhead di misura; non è il tempo isolato del kernel. La stampa/scansione
dell'output **non è normalmente eseguita**: con il default di PolyBench
`polybench_prevent_dce` la condiziona ad argomenti particolari, assenti nella
chiamata dello script. La stampa sarebbe invece inclusa se la compilazione
abilitasse `POLYBENCH_DUMP_ARRAYS`; i flag effettivi non sono conservati nel
CSV. `task_clock_ms` è tempo CPU aggregato, non tempo trascorso. Cachegrind
simula gli accessi e altera tempi e scheduling: i suoi tempi wall non vengono
usati per speedup nativi.

La mediana delle cinque ripetizioni è usata come valore centrale perché è meno
sensibile di una media a singole prove anomale; gli intervalli min–max rendono
visibile la variabilità, ma con cinque campioni non costituiscono un test di
significatività. Le matrici piccole sono escluse dai grafici temporali del
confronto principale: i tempi misurati sono vicini all'overhead di processo e
alla risoluzione millisecondo del campo wall. Compaiono invece nel grafico
relativo ai dataset, da interpretare con cautela.

| Aspetto | Evidenza e limite |
|---|---|
| Copertura `perf` | 450 righe: 9 versioni × 5 dimensioni × 2 richieste thread × 5 ripetizioni. |
| Copertura Cachegrind | 71 righe su 90 combinazioni teoriche. Mancano tutte le combinazioni `N=4000` (18) e `v2_restrict`, `N=2000`, 8 thread (1). |
| Campagne | v0, v2, v3 e v4 hanno 200 righe `perf` e 32 Cachegrind; le cinque varianti intermedie hanno 250 righe `perf` e 39 Cachegrind. Il secondo gruppo è stato aggiunto successivamente. |
| Stato dei comandi | Le righe misurate riportano stato di uscita zero; questo non prova correttezza numerica né disponibilità/semantica di tutti i contatori hardware. |
| Ambiente | Le note nello script indicano Nano1 e suggeriscono `OMP_DYNAMIC=FALSE OMP_PROC_BIND=TRUE OMP_PLACES=cores`. Il CSV non registra CPU, core disponibili, sistema/kernel effettivi, compilatore/versione, flag effettivi, affinità, carico o data: questi dettagli e l'uso effettivo delle variabili restano da confermare. |

Il protocollo sceglie più dimensioni per osservare la crescita del lavoro e
due quantità di thread per vedere se il parallelismo richiesto aiuta; mediana e
intervallo evitano di basarsi su una singola esecuzione. Le scelte non
garantiscono che 4 e 8 siano confronti equi: senza core disponibili, affinità
e dimensione effettiva del team, non si può escludere oversubscription.
Inoltre, i risultati originali e quelli intermedi provengono da campagne
separate. Senza metadati non si può verificare la comparabilità ambientale:
la tabella dei passaggi e i grafici che attraversano quel confine vanno letti
con cautela.

## 3. Originale, pragma OpenMP e SIMD

Il grafico usa le cinque mediane per configurazione e mostra anche le versioni
finali. L'originale è **seriale**: le etichette 4 e 8 indicano gruppi distinti
di prove in cui era impostato `OMP_NUM_THREADS`, non una parallelizzazione
dell'originale. Le barre riportano tempi assoluti, non IPC.

![Tempi dell'originale, OpenMP, SIMD, static e static,1 per N=4000](./grafici_presentazione/wall_time_simd_static_static1_n4000.svg)

| N | Thread richiesti | Originale `v0` (s) | Pragma `v2` (s) | `v2_simd` (s) | Originale / pragma | Originale / SIMD |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 4 | 1.093 | 0.629 | 1.112 | 1.74× | 0.98× |
| 1024 | 8 | 1.093 | 0.720 | 1.269 | 1.52× | 0.86× |
| 2000 | 4 | 7.587 | 4.797 | 8.317 | 1.58× | 0.91× |
| 2000 | 8 | 7.587 | 5.033 | 8.308 | 1.51× | 0.91× |
| 4000 | 4 | 60.509 | 39.301 | 54.055 | 1.54× | 1.12× |
| 4000 | 8 | 60.585 | 40.140 | 53.058 | 1.51× | 1.14× |

Ogni rapporto è la **mediana originale divisa per la mediana della versione
indicata**: un valore maggiore di 1 indica un tempo osservato inferiore.
Per `N=4000`, i rapporti `v2 / v2_simd` sono 0.727 e 0.757: la versione SIMD
ha mediane più alte di quella con soli pragma. Il risultato è descrittivo:
originale e pragma appartengono alla campagna iniziale, mentre SIMD è nella
campagna successiva. Senza ambiente, flag effettivi e ordine delle prove non
si può concludere che le direttive SIMD abbiano causato il peggioramento.

Per una lettura delle differenze, a `N=4000` il gruppo 4 thread ha intervalli
originale [60.469–60.549] s, OpenMP [39.283–40.832] s e SIMD
[53.781–54.298] s; nel gruppo 8 thread sono rispettivamente
[60.535–60.613], [39.993–40.500] e [52.911–53.244] s. Sono minimi e
massimi di cinque prove, non intervalli di confidenza.

## 4. Confronto principale: `v2_simd`, static e static,1

I grafici seguenti si concentrano su `N >= 1024`, dove il lavoro è più
significativo rispetto all'overhead. Le barre verticali indicano min–max delle
cinque prove; la scala dei tempi è logaritmica.

![Tempi mediani di v2_simd, v3 e v4 con 4 thread](./lu_results_intermedi_plots/confronto_simd_v3_v4_t4.svg)

![Tempi mediani di v2_simd, v3 e v4 con 8 thread](./lu_results_intermedi_plots/confronto_simd_v3_v4_t8.svg)

| N | Thread | `v2_simd` mediana [min–max] s | v3 mediana [min–max] s | v4 mediana [min–max] s | `v2_simd` / v3 | v3 / v4 |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 4 | 1.112 [1.090–1.216] | 0.595 [0.572–0.616] | 0.618 [0.598–0.638] | 1.87 | 0.963 |
| 1024 | 8 | 1.269 [1.175–1.360] | 0.704 [0.662–0.738] | 0.743 [0.702–0.748] | 1.80 | 0.948 |
| 2000 | 4 | 8.317 [8.254–8.338] | 4.707 [4.676–4.719] | 4.672 [4.653–4.709] | 1.77 | 1.007 |
| 2000 | 8 | 8.308 [8.242–8.529] | 4.920 [4.903–4.982] | 4.926 [4.890–4.944] | 1.69 | 0.999 |
| 4000 | 4 | 54.055 [53.781–54.298] | 38.566 [38.505–39.537] | 38.536 [38.499–39.676] | 1.40 | 1.001 |
| 4000 | 8 | 53.058 [52.911–53.244] | 39.250 [39.165–39.871] | 39.188 [39.067–39.311] | 1.35 | 1.002 |

I rapporti sono fra mediane: `v2_simd / v3 > 1` significa che la mediana di
v3 è minore; `v3 / v4 > 1` significa che v4 ha mediana minore. La mediana di
v3 è inferiore a quella di `v2_simd` in tutti i casi mostrati, di un fattore
circa 1.35–1.87. È però un confronto tra la variante SIMD isolata e una
versione con più modifiche, e attraversa le due campagne: non misura il
contributo della sola SIMD né esclude differenze di macchina o condizioni.

Fra v3 e v4 le mediane sono vicine: il rapporto varia da 0.948 a 1.007. Gli
intervalli min–max si sovrappongono in tutti i gruppi riportati, pur senza
costituire un test statistico; con sole cinque prove questi dati non
supportano una conclusione robusta che `static` o `static,1` sia migliore.
Per `N=4000`, ad esempio, le mediane differiscono di circa 0.08% con
4 thread richiesti e 0.16% con 8, molto meno dell'ampiezza degli intervalli.
Con `N=1024`, invece, v4 ha una mediana più alta del 3.9% (4 thread) e del
5.5% (8 thread), ma anche qui gli intervalli si sovrappongono.

## 5. Effetto dei singoli passaggi

I grafici a `N=4000` mostrano la sequenza cumulativa. Le interruzioni tra i
gruppi evidenziano che v2 e le varianti intermedie, così come `v2_restrict` e
v3, non sono state misurate nella stessa campagna.

![Passaggi cumulativi, N=4000 e 4 thread](./lu_results_intermedi_plots/passaggi_ottimizzazione_n4000_t4.svg)

![Passaggi cumulativi, N=4000 e 8 thread](./lu_results_intermedi_plots/passaggi_ottimizzazione_n4000_t8.svg)

La colonna rapporto è `mediana precedente / mediana successiva`: sopra 1
indica una mediana inferiore dopo il passaggio. I rapporti non sono speedup
statisticamente validati.

| Passaggio | 4 thread: mediane (s), rapporto | 8 thread: mediane (s), rapporto | Lettura |
|---|---:|---:|---|
| v2 → `v2_simd` | 39.301 → 54.055; **0.727** | 40.140 → 53.058; **0.757** | `v2_simd` non risulta più veloce in questo CSV; confronto fra campagne. |
| `v2_simd` → `v2_recip` | 54.055 → 53.894; **1.003** | 53.058 → 53.136; **0.999** | Differenza trascurabile rispetto alla variabilità. |
| `v2_recip` → `v2_hoist` | 53.894 → 53.412; **1.009** | 53.136 → 52.768; **1.007** | Piccolo calo della mediana; effetto non isolabile dal rumore/campagna. |
| `v2_hoist` → `v2_rows` | 53.412 → 53.604; **0.996** | 52.768 → 52.576; **1.004** | Nessun andamento consistente tra i due valori di thread. |
| `v2_rows` → `v2_restrict` | 53.604 → 53.570; **1.001** | 52.576 → 52.660; **0.998** | Mediane praticamente uguali. |
| `v2_restrict` → v3 (`static`) | 53.570 → 38.566; **1.389** | 52.660 → 39.250; **1.342** | Salto marcato, ma le misure sono di campagne diverse: non attribuirlo al solo `static`. |
| v3 (`static`) → v4 (`static,1`) | 38.566 → 38.536; **1.001** | 39.250 → 39.188; **1.002** | Differenza minima, con intervalli sovrapposti. |

Per i passaggi intermedi da `v2_simd` a `v2_restrict`, le mediane a
`N=4000` restano circa fra 53 e 54 secondi. I miglioramenti o peggioramenti
fra passaggi sono per lo più inferiori all'1%, senza un andamento uniforme
tra 4 e 8 thread. La sola presenza di direttive SIMD o di `restrict` non
garantisce un miglioramento: servirebbero condizioni di misura controllate e
il report di vettorizzazione del compilatore per capire quale codice è stato
effettivamente generato. In particolare, il confronto `v2_restrict` → v3
include più differenze di codice e un cambio di campagna, quindi non consente
di attribuire il risultato allo scheduling.

### 5.1 IPC, istruzioni e cicli: che cosa cambia davvero

L'IPC (*instructions per cycle*) è il rapporto tra istruzioni ritirate e
cicli contati. Per ogni ripetizione si calcola `instructions / cycles`, poi
si prende la mediana dei cinque rapporti. Nel confronto di sotto, invece,
istruzioni e cicli sono **mediane separate** dei rispettivi contatori. Le
misure riguardano il processo, inclusi il runtime OpenMP e i suoi thread:
non sono l'IPC isolato del ciclo interno e possono risentire di attese,
affinità e multiplexing.

![IPC per l'originale e i passaggi successivi, N=4000](./grafici_presentazione/ipc_n4000.svg)

| Versione, N=4000 | IPC 4 richiesti | Istruzioni 4 (miliardi) | Cicli 4 (miliardi) | IPC 8 richiesti | Istruzioni 8 (miliardi) | Cicli 8 (miliardi) |
|---|---:|---:|---:|---:|---:|---:|
| Originale `v0` (seriale) | 1.911 | 170.969 | 89.444 | 1.909 | 170.969 | 89.558 |
| Pragma `v2` | 0.775 | 175.791 | 226.815 | 0.759 | 171.714 | 226.172 |
| SIMD `v2_simd` | 0.598 | 121.174 | 202.711 | 0.501 | 107.878 | 215.157 |
| Reciproco | 0.598 | 121.154 | 202.587 | 0.501 | 107.873 | 215.422 |
| Hoisting | 0.496 | 99.778 | 201.088 | 0.405 | 86.562 | 213.678 |
| Puntatori alle righe | 0.496 | 99.762 | 201.240 | 0.405 | 86.554 | 213.744 |
| `restrict` | 0.496 | 99.657 | 201.168 | 0.405 | 86.554 | 213.607 |
| `static` | 0.416 | 91.730 | 220.552 | 0.393 | 86.456 | 219.959 |
| `static,1` | 0.415 | 91.462 | 220.177 | 0.393 | 86.443 | 220.015 |

L'originale ha IPC più alto, ma la variante con pragma ha tempo trascorso
inferiore: **un IPC maggiore non equivale automaticamente a prestazioni
migliori**. La versione parallela somma lavoro e cicli dei thread, inclusi
quelli del runtime, mentre più thread possono progredire contemporaneamente.
Inoltre un'istruzione vettoriale può eseguire più operazioni utili: contare
istruzioni non equivale a contare operazioni o lavoro matematico.

Tra SIMD e hoisting, le mediane delle istruzioni diminuiscono da 121.174 a
99.778 miliardi con 4 thread richiesti (**−17.7%**) e da 107.878 a 86.562
miliardi con 8 (**−19.8%**), mentre le mediane dei cicli diminuiscono di circa
0.8% e 0.7%. Di conseguenza anche l'IPC scende: è coerente con meno
istruzioni ritirate su un numero di cicli simile, non dimostra un peggioramento
del tempo. Questi sono confronti fra stadi cumulativi della campagna intermedia;
non dimostrano da soli che il risparmio sia interamente causato dall'hoisting.
Le mediane wall corrispondenti passano da 54.055 a 53.412 s e da 53.058 a
52.768 s: l'effetto sul tempo osservato è molto più piccolo del calo delle
istruzioni. Il grafico interrompe le linee ai cambi di campagna.

Tra `static` e `static,1`, a `N=4000`, gli IPC mediani sono rispettivamente
0.416 e 0.415 con 4 thread e circa 0.393 per entrambe con 8. Anche i tempi sono
molto vicini. Non si può dedurre un vantaggio robusto di scheduling dai soli
contatori o da cinque ripetizioni. A `N=1024` la differenza mediana di tempo
è più evidente, ma i range osservati si sovrappongono.

![Differenza percentuale tra static,1 e static nei dataset](./grafici_presentazione/static_vs_static1_per_dataset.svg)

## 6. Interpretazione, correttezza e limiti

- I tempi per `N=32` e `N=128` sono dominati in larga misura da avvio,
  inizializzazione e overhead del processo: non sono una base affidabile per
  decidere quale kernel sia più veloce.
- Per `N=1024`, 2000 e 4000 le mediane con 4 thread richiesti sono inferiori
  a quelle con 8 per `v2`, `static` e `static,1`; non vale per tutte le
  varianti intermedie. Il CSV non permette di stabilire se la causa sia
  saturazione, core disponibili, affinità, frequenze o sovraccarico.
- I contatori `perf` descrivono il processo e possono includere il runtime
  OpenMP; permessi, supporto e multiplexing degli eventi dipendono dalla
  macchina. Non sono una misura diretta di vettorizzazione o bilanciamento.
- La matrice iniziale è `A[i][j] = (i+1)(j+1)/N`, quindi ha rango uno. Per
  `N>1` è singolare e LU senza pivoting può incontrare pivot nulli o
  numericamente inaffidabili. `exit_status=0` non assicura valori finiti né
  una fattorizzazione corretta.

Prima di usare il benchmark come valutazione delle prestazioni di una LU
corretta, occorre usare un input non singolare adatto all'assenza di pivoting,
verificare valori finiti e residuo/ricostruzione rispetto a un riferimento e
ripetere l'intera campagna con lo stesso input. Occorre anche registrare
macchina, compilatore e flag, numero di core e team effettivo, affinità,
frequenze/carico, ordine delle prove e identificativo delle sorgenti. Una
campagna unica con esecuzioni intercalate o randomizzate può ridurre gli
effetti dell'ordine; occorre controllare anche il multiplexing di `perf`.


## 7. Conclusioni

I passaggi intermedi hanno effetti wall piccoli e non uniformi; l'IPC diminuisce insieme al numero
di istruzioni e non è, da solo, indice di velocità. Le versioni `static` e
`static,1` hanno tempi e IPC vicini a `N=4000`; la variabilità e la copertura
disponibile non sostengono una scelta definitiva dello scheduling.
