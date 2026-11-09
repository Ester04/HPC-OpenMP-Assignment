# Guida: ottimizzare la fattorizzazione LU con OpenMP (partendo da zero)

Questa guida spiega, senza dare nulla per scontato, che cosa fa il codice di `lu.c`, come funziona OpenMP e perché ogni versione (0-4) è più veloce, o almeno diversa, dalla precedente. La **versione 2** ha una sezione molto estesa, perché è quella dove si capisce davvero come ragiona OpenMP.

Indice:

1. Cos'è OpenMP e come funziona
2. Il problema: dove si può parallelizzare la LU
3. Versione 0: la baseline seriale
4. Versione 1: `parallel for` sui cicli interni
5. **Versione 2: una sola regione parallela (spiegazione approfondita)**
6. Versione 3: micro-ottimizzazioni (reciproco, hoisting, `restrict`, SIMD)
7. Versione 4: scheduling `static` contro `static,1`
8. Come compilare, eseguire e leggere i risultati
9. Errori tipici e glossario

---

## 1. Cos'è OpenMP e come funziona

### 1.1 Thread e memoria condivisa

Un programma normale esegue le istruzioni una dopo l'altra, in un solo **thread** (un "filo" di esecuzione). Una CPU moderna ha più **core**, ognuno capace di eseguire un thread alla volta. Se il programma usa un solo thread, gli altri core restano inattivi.

Con più thread nello stesso programma, **tutti vedono la stessa memoria**: se un thread scrive `A[3][4]`, un altro thread che lo legge poco dopo vede il valore nuovo. Questo è comodo, ma è anche la fonte di tutti i problemi: se due thread toccano la stessa cella senza coordinarsi, il risultato dipende da chi arriva prima (vedi §1.5).

### 1.2 OpenMP in una frase

**OpenMP** è un insieme di direttive (`#pragma omp ...`) che si scrivono nel codice C/C++ per dire al compilatore: "questa parte va eseguita da più thread". Non cambia il linguaggio: se il compilatore non conosce OpenMP, ignora i `#pragma` e il programma resta seriale e corretto. Per attivarlo si compila con `-fopenmp` (gcc/clang).

Esempio minimo:

```c
#pragma omp parallel for
for (int i = 0; i < 1000; i++)
    b[i] = 2 * a[i];
```

OpenMP divide le 1000 iterazioni tra i thread (per esempio 250 ciascuno con 4 thread) e le esegue in contemporanea. Qui funziona perché ogni iterazione scrive una cella diversa (`b[i]`) e non dipende dalle altre.

### 1.3 Il modello fork-join

OpenMP lavora così:

```
 thread master ─────────┬──────────────────────┬────────── (prosegue da solo)
                        │  FORK                │  JOIN
                        ├─ thread 0 ───────────┤
                        ├─ thread 1 ───────────┤   regione parallela
                        ├─ thread 2 ───────────┤
                        └─ thread 3 ───────────┘
```

- Il programma parte con un solo thread (il *master*).
- Quando incontra una **regione parallela** (`#pragma omp parallel`) si ha il **fork**: nasce una squadra (*team*) di thread.
- Alla fine della regione c'è il **join**: tutti i thread devono aver finito, poi resta solo il master.

Quanti thread? Lo decide la variabile d'ambiente `OMP_NUM_THREADS` (per esempio `OMP_NUM_THREADS=8 ./lu_v2`).

### 1.4 Variabili `shared` e `private`

Dentro una regione parallela ogni variabile è di uno di due tipi:

- **shared** (condivisa): esiste una sola copia, tutti i thread vedono e modificano la stessa. Per default lo sono le variabili dichiarate *prima* della regione (per esempio la matrice `a` e la dimensione `n`).
- **private**: ogni thread ha la propria copia, indipendente dalle altre. Per default lo sono le variabili dichiarate *dentro* la regione.

Con `private(j)` si dice esplicitamente "ogni thread abbia il suo `j`". È fondamentale per gli indici dei cicli: se due thread usassero lo stesso `j` condiviso, si sovrascriverebbero a vicenda il contatore.

### 1.5 Data race e barriere

Una **data race** si ha quando due thread accedono alla stessa cella di memoria senza sincronizzazione e almeno uno dei due scrive. Il risultato dipende dall'ordine casuale con cui i thread arrivano: il programma può dare risultati sbagliati *a volte sì e a volte no*, che è il peggior tipo di bug da trovare.

Una **barriera** (*barrier*) è un punto d'incontro: ogni thread che la raggiunge si ferma finché **tutti** gli altri thread del team non l'hanno raggiunta. Solo allora tutti ripartono. Le barriere servono a imporre un ordine tra fasi: "nessuno inizi la fase B finché tutti non hanno finito la fase A".

Prezzo delle barriere: i thread veloci aspettano quelli lenti, e il meccanismo stesso di sincronizzazione ha un costo. Troppe barriere con poco lavoro in mezzo rallentano il programma.

### 1.6 Il costrutto `omp for`

`#pragma omp for` **distribuisce le iterazioni** di un ciclo tra i thread del team. Si può scrivere solo dentro una regione parallela già aperta. La scorciatoia `#pragma omp parallel for` è la combinazione dei due: apre la regione e distribuisce il ciclo.

Due regole importanti:

1. L'indice del ciclo che segue `omp for` è automaticamente privato.
2. Alla fine di un `omp for` c'è una **barriera implicita**: nessuno prosegue finché tutte le iterazioni, di tutti i thread, non sono finite. Si può togliere con `nowait`, ma solo se le istruzioni successive non dipendono dai risultati del ciclo.

---

## 2. Il problema: dove si può parallelizzare la LU

### 2.1 Che cosa calcola il codice

La fattorizzazione LU riscrive la matrice `A` (quadrata, `n × n`) in modo che contenga due matrici triangolari `L` e `U` con `A = L · U`. Il codice lavora **in place**: i risultati sostituiscono i valori originali. Si procede per passi `k = 0, 1, ..., n-1`. Al passo `k`:

**Fase 1: normalizzazione della riga `k`.** Si divide ogni elemento a destra del pivot `A(k,k)` per il pivot stesso:

```c
for (j = k + 1; j < n; j++)
    A(k, j) = A(k, j) / A(k, k);
```

**Fase 2: aggiornamento del blocco in basso a destra.** Per ogni riga `i` sotto la `k` e ogni colonna `j` a destra della `k`:

```c
for (i = k + 1; i < n; i++)
    for (j = k + 1; j < n; j++)
        A(i, j) = A(i, j) - A(i, k) * A(k, j);
```

Dopo il passo `k`, la riga `k` e la colonna `k` sono "finite" e non vengono più toccate. Il lavoro si restringe al blocco più piccolo `(k+1..n-1) × (k+1..n-1)`.

(`A(i,j)` nel codice è una macro per `a[i*ld + j]`: la matrice è memorizzata in un unico vettore, riga dopo riga.)

### 2.2 Cosa si può eseguire in parallelo e cosa no

Si ragiona chiedendosi: "quali iterazioni sono indipendenti tra loro?"

| Ciclo | Parallelizzabile? | Perché |
|---|---|---|
| Ciclo esterno su `k` | **No** | Il passo `k+1` usa i risultati finali del passo `k` (la riga `k+1` aggiornata e il nuovo pivot). C'è una dipendenza vera. |
| Fase 1 (ciclo su `j`) | **Sì** | Ogni iterazione scrive una cella diversa `A(k,j)` e legge solo `A(k,k)`, che nessuno modifica (`j` parte da `k+1`). |
| Fase 2 (ciclo su `i`) | **Sì** | L'iterazione `i` scrive solo la riga `i`. Legge la riga `k` e `A(i,k)`, che in questa fase non vengono scritte da nessuno. Righe diverse non interferiscono. |

Conclusione: il ciclo `k` resta seriale, e si parallelizza **dentro** ogni passo. Ma c'è un ordine obbligato:

```
  passo k:   [ fase 1 ]  →  [ fase 2 ]  →  passo k+1:  [ fase 1 ]  →  [ fase 2 ]  → ...
              parallela      parallela                    parallela      parallela
                       ↑            ↑
              queste frecce sono le dipendenze: richiedono una barriera
```

Tutta la storia delle versioni 1 e 2 riguarda **come** si realizzano queste frecce.

---

## 3. Versione 0: la baseline seriale

```c
for (k = 0; k < n; k++) {
    for (j = k + 1; j < n; j++)
        A(k, j) = A(k, j) / A(k, k);
    for (i = k + 1; i < n; i++)
        for (j = k + 1; j < n; j++)
            A(i, j) = A(i, j) - A(i, k) * A(k, j);
}
```

È il codice originale, senza OpenMP. Serve come **riferimento**:

- per la correttezza (le altre versioni devono dare lo stesso risultato, a meno di errori di arrotondamento);
- per misurare lo *speedup* (quante volte è più veloce una versione rispetto a questa).

Costo: circa `(2/3)·n³` operazioni in virgola mobile. Per `n = 2000` sono circa 5,3 miliardi di operazioni.

---

## 4. Versione 1: `parallel for` sui cicli interni

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

### Cosa succede

Il ciclo `k` è fuori da qualsiasi regione parallela: lo esegue il solo thread master. A ogni passo `k` si incontrano **due regioni parallele**, una per fase. Ciascuna ha un fork all'inizio e un join (con barriera) alla fine.

```
 master:  k=0 ┬─fase1─┬ ┬─fase2─┬  k=1 ┬─fase1─┬ ┬─fase2─┬  k=2 ...
              fork  join fork  join      fork  join fork  join
```

### Perché è corretta

- Il join di ogni regione fa da barriera: la fase 2 parte solo dopo che la fase 1 è finita per tutti, e il passo `k+1` parte solo dopo la fase 2 del passo `k`. Le dipendenze della §2.2 sono rispettate "gratis".
- Le iterazioni distribuite sono indipendenti (§2.2), quindi non ci sono data race.

### Il dettaglio di `private(j)`

Nel secondo ciclo, `i` è l'indice del ciclo parallelo e diventa automaticamente privato. Ma `j` è l'indice di un ciclo **interno**, dichiarato fuori dalla regione (`int i, j, k;` in cima alla funzione): per default sarebbe **shared**. Tutti i thread aggiornerebbero lo stesso `j` e si romperebbero i cicli a vicenda. `private(j)` risolve il problema.

(Nella prima regione non serve: lì `j` è proprio l'indice del ciclo parallelo.)

### Il limite

Per ogni passo `k` ci sono due fork-join: in totale `2n` regioni parallele. Quando `k` è grande il lavoro rimasto è poco (il blocco è piccolo), ma il costo fisso di avviare e chiudere una regione parallela resta lo stesso. Il tempo si va a perdere nell'organizzazione, non nei calcoli. Questo è il problema che risolve la versione 2.

---

## 5. Versione 2: una sola regione parallela (spiegazione approfondita)

```c
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
```

Questa è la parte che si capisce meglio con calma. Il punto da afferrare è che **il ciclo `k` ora è dentro la regione parallela**, e questo cambia il significato di tutto.

### 5.1 L'idea in un'immagine

Un'analogia: una squadra di 4 operai deve dipingere una stanza in `n` fasi, ciascuna suddivisa in due sotto-fasi che richiedono il lavoro di tutti.

- **Versione 1:** a ogni sotto-fase si chiama la squadra, la si fa entrare, si dà il compito e si fa uscire. Per `2n` sotto-fasi si ripetono `2n` volte la chiamata, l'ingresso e l'uscita.
- **Versione 2:** la squadra entra **una volta sola** e resta nella stanza. Ognuno ha una copia dello stesso copione ("per ogni fase k: sotto-fase 1, poi sotto-fase 2") e lo segue per conto suo. Ai punti d'incontro previsti dal copione (le barriere) si aspettano a vicenda.

### 5.2 Cosa fa `#pragma omp parallel` da solo

Questo è il passaggio meno intuitivo. Scritto **senza** `for`:

```c
#pragma omp parallel
{
    // ... codice ...
}
```

il blocco **non viene diviso tra i thread: viene eseguito per intero da ciascun thread**. Con 4 thread, ogni istruzione del blocco viene eseguita 4 volte, una per thread, in parallelo. Si chiama modello *SPMD* (Single Program, Multiple Data): lo stesso programma, eseguito da tutti.

Nella versione 2 il "blocco" è l'intero ciclo `for (k ...)`. Quindi:

- **ognuno dei 4 thread esegue il proprio ciclo `k`**, da 0 a `n-1`;
- per questo `k` deve essere `private`: ogni thread ha il proprio contatore. Se `k` fosse condiviso, un thread che fa `k++` cambierebbe il `k` degli altri;
- il ciclo `k` in sé non è "diviso": ogni thread lo percorre tutto. Il lavoro vero lo divide `omp for` all'interno.

Dettaglio tecnico: l'indice di un ciclo che segue `omp for` è automaticamente privato, ma `k` è l'indice di un ciclo *normale*, non preceduto da `omp for`. Va quindi dichiarato `private` a mano. Per la stessa ragione servono `i` e `j`. (Nella versione 3 gli indici sono dichiarati *dentro* il `for`, con `for (int k = ...)`, e sono automaticamente privati.)

### 5.3 Cosa fa `#pragma omp for` dentro la regione

Quando ogni thread, seguendo il suo ciclo `k`, incontra:

```c
#pragma omp for
for (j = k + 1; j < n; j++)
    ...
```

OpenMP sa che tutti i thread stanno arrivando allo stesso ciclo, quindi **divide le iterazioni** `j = k+1 ... n-1` tra di loro. Ciascun thread esegue solo la propria fetta. Per default (schedule `static`) le iterazioni sono divise in blocchi contigui di dimensioni quasi uguali, uno per thread.

Esempio con `n = 6`, `k = 0`, 2 thread: le iterazioni `j = 1..5` sono 5. Il thread 0 prende `j = 1, 2, 3` e il thread 1 prende `j = 4, 5`.

### 5.4 Traccia passo per passo

Prendiamo `n = 5` e 2 thread. In ogni riga si vede cosa fa ciascun thread.

| Istante | Thread 0 | Thread 1 |
|---|---|---|
| `k = 0`, fase 1 | normalizza `A(0,1)`, `A(0,2)` | normalizza `A(0,3)`, `A(0,4)` |
| **barriera B1** | aspetta | aspetta |
| `k = 0`, fase 2 | aggiorna le righe `i = 1, 2` | aggiorna le righe `i = 3, 4` |
| **barriera B2** | aspetta | aspetta |
| `k = 1`, fase 1 | normalizza `A(1,2)`, `A(1,3)` | normalizza `A(1,4)` |
| **barriera B1** | ... | ... |
| `k = 1`, fase 2 | aggiorna le righe `i = 2, 3` | aggiorna la riga `i = 4` |
| **barriera B2** | ... | ... |
| `k = 2`, ... | ... | ... |

Si osserva che:

- i due thread eseguono entrambi **tutto il ciclo `k`**, ma si dividono le iterazioni interne;
- a ogni passo ci sono due barriere; non nascono e non muoiono thread;
- finché `k` aumenta, le fette si rimpiccioliscono (il blocco residuo si restringe).

### 5.5 Perché i thread restano allineati

Una domanda legittima: se ogni thread ha il suo `k` privato, come si evita che un thread sia al passo `k = 3` mentre un altro è ancora al `k = 1`? Per due motivi:

1. **Le barriere.** Nessun thread supera una barriera finché tutti non l'hanno raggiunta. Quindi i thread possono differire al massimo di "mezzo passo" (uno è ancora nella fase 2 mentre un altro è già arrivato alla barriera).
2. **Il ciclo è deterministico.** I limiti (`k = 0 .. n-1`) non dipendono da dati che i thread modificano, quindi tutti percorrono esattamente la stessa sequenza di valori di `k`. Di conseguenza tutti incontrano gli stessi `omp for` e le stesse barriere nello stesso ordine.

Questa seconda condizione è una regola di OpenMP: **ogni costrutto di worksharing (`omp for`) e ogni barriera devono essere incontrati da tutti i thread del team, nello stesso ordine**. Se un thread saltasse un `omp for` che gli altri eseguono, il programma si bloccherebbe o darebbe risultati scorretti. Qui la regola è rispettata perché non ci sono `if` che dipendano dal thread.

### 5.6 Perché servono entrambe le barriere (e perché non si può usare `nowait`)

Al termine di ogni `omp for` c'è una barriera implicita. Si potrebbe pensare di toglierle con `nowait` per guadagnare tempo: **sarebbe un errore**. Vediamo cosa succederebbe.

**Barriera B1 (tra fase 1 e fase 2).** La fase 2 legge `A(k,j)`, cioè la riga `k` appena normalizzata dalla fase 1. Se togliessimo la barriera:

```
 Thread 0: sta ancora normalizzando A(0,2)    ← lento
 Thread 1: ha finito A(0,3), A(0,4), passa alla fase 2:
           aggiorna la riga 3 usando A(0,2)   ← valore NON ancora normalizzato
```

Il thread 1 usa un valore vecchio. Il risultato è sbagliato **in silenzio**: nessun errore, nessun crash, solo numeri errati, e magari corretti in un'altra esecuzione in cui i thread hanno avuto una temporizzazione diversa. Questa è la tipica data race.

**Barriera B2 (tra fase 2 del passo `k` e fase 1 del passo `k+1`).** Al passo `k+1` la fase 1 legge il nuovo pivot `A(k+1,k+1)` e modifica la riga `k+1`, che nel passo `k` è stata aggiornata dalla fase 2 da *un solo thread* (quello a cui è toccata la riga `i = k+1`). Senza barriera, un altro thread, già in fase 1 del passo `k+1`, potrebbe leggere `A(k+1,k+1)` prima che il suo proprietario l'abbia aggiornato.

Non sono due barriere "di cortesia", ma **le due dipendenze vere** della §2.2, tradotte in sincronizzazione. Non si possono togliere.

### 5.7 Cosa si guadagna rispetto alla versione 1

| | Versione 1 | Versione 2 |
|---|---|---|
| Regioni parallele | `2n` | 1 |
| Dispatch del lavoro | a ogni regione il master deve avviare/risvegliare i thread e assegnare il lavoro | una volta sola, poi i thread procedono da soli |
| Barriere | `2n` (nei join) | `2n` (implicite negli `omp for`) |

Il numero di sincronizzazioni è lo stesso. Si risparmia il costo di **aprire** una regione parallela `2n` volte: preparare il team, risvegliare i thread, distribuire le informazioni. Una nota di onestà: le implementazioni moderne (come libgomp di gcc) non creano e distruggono davvero i thread a ogni regione, ma li tengono in un *pool* e li risvegliano. Il costo del risveglio è comunque più alto del semplice passaggio da una barriera all'altra.

Quanto si guadagna in pratica dipende dalla macchina. Nel test fatto durante la preparazione dei file (2 core, `n = 1200`) la differenza tra versione 1 e 2 era trascurabile (speedup 2,05 contro 1,94). Con molti core, o con `n` piccolo, in cui il lavoro per passo è poco, la differenza cresce. Il passo 2 è la base per le ottimizzazioni successive, che richiedono di trovarsi dentro una regione parallela unica.

### 5.8 Cosa si poteva sbagliare

- **Dimenticare `private(k, i, j)`:** gli indici sarebbero condivisi e i thread si pesterebbero i piedi. Il bug è subdolo perché con 1 thread funziona.
- **Mettere codice "libero" dentro la regione:** per esempio `contatore++` con `contatore` condiviso. Verrebbe eseguito da tutti i thread (5.2), con una data race. Per codice che deve girare una volta sola si usa `#pragma omp single` (un solo thread, qualunque) o `omp master` (il thread 0).
- **Usare `nowait`:** vedi 5.6.

---

## 6. Versione 3: micro-ottimizzazioni

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

La struttura parallela è identica alla versione 2. Cambia il corpo dei cicli, per far lavorare meglio **ogni singolo core**. Quattro modifiche:

### 6.1 Reciproco invece di divisione

Prima: `A(k,j) = A(k,j) / A(k,k)` per ogni `j`, cioè `n-k-1` divisioni. Ora si calcola **una** volta `inv = 1.0 / A(k,k)` e poi si moltiplica.

Perché: in un processore una divisione in virgola mobile impiega diversi cicli di clock (nell'ordine di 10-20), una moltiplicazione pochi (circa 4) e può essere eseguita in pipeline. Sostituire molte divisioni con una divisione e molte moltiplicazioni è un classico.

Attenzione: `x / p` e `x * (1/p)` possono differire nell'ultimo bit per via dell'arrotondamento. È per questo che nelle versioni 3 e 4 la differenza rispetto alla baseline è di circa `3e-13` e non esattamente zero. È normale, ed è il motivo per cui il controllo di correttezza usa una **tolleranza** e non un confronto esatto.

### 6.2 Hoisting di `A(i,k)`

```c
const double aik = A(i, k);
```

`A(i,k)` non dipende da `j`: è lo stesso per tutta la riga. Portarlo in una variabile fuori dal ciclo interno (*hoisting*, "sollevamento") lo mantiene in un registro del processore, invece di rileggerlo dalla memoria a ogni iterazione. Il compilatore a volte non lo fa da solo, perché non è sicuro che le scritture su `A(i,j)` non cambino anche `A(i,k)`.

### 6.3 `restrict`

```c
double *restrict Ai = &A(i, 0);
const double *restrict Ak = &A(k, 0);
```

Il compilatore, vedendo due puntatori, deve pensare al caso peggiore: "e se puntassero alla stessa zona di memoria (*aliasing*)?". In quel caso non potrebbe riordinare né vettorizzare le operazioni. `restrict` è una promessa del programmatore: "durante la vita di questo puntatore, quella memoria si accede solo attraverso di lui". Qui è vero, perché le righe `i` e `k` sono diverse (`i > k`). Con questa informazione il compilatore ottimizza di più.

### 6.4 SIMD (`omp simd`)

I processori moderni hanno **registri vettoriali** che contengono più numeri insieme (4 `double` con AVX2, 8 con AVX-512) ed eseguono la stessa operazione su tutti con **una sola istruzione**. Si chiama *SIMD* (Single Instruction, Multiple Data).

Il ciclo interno è ideale: legge e scrive celle consecutive (`Ai[j]`, `Ak[j]`) e ogni iterazione fa la stessa operazione.

`#pragma omp simd` dice al compilatore: "vettorizza questo ciclo; ti garantisco che le iterazioni sono indipendenti". Si abbina a `-march=native`, che permette di usare le istruzioni vettoriali più recenti della tua CPU, comprese le **FMA** (*fused multiply-add*: `a*b + c` in una sola istruzione, con un solo arrotondamento, altra causa di piccole differenze numeriche).

Nella prima fase si usa `omp for simd`: prima distribuisce le iterazioni tra i thread, poi vettorizza la porzione di ciascun thread. Sono due livelli di parallelismo diversi che si compongono: i **thread** (più core) e i **vettori** (più dati per istruzione in un core).

### 6.5 Riassunto

| Modifica | Problema risolto |
|---|---|
| Reciproco | Divisioni costose |
| Hoisting `aik` | Riletture inutili dalla memoria |
| `restrict` | Il compilatore teme l'aliasing e non ottimizza |
| `omp simd` | Una sola operazione per volta invece di 4-8 |

---

## 7. Versione 4: scheduling `static` contro `static,1`

Cambia una sola riga rispetto alla versione 3:

```c
#pragma omp for schedule(static, 1)    // versione 4
// al posto di
#pragma omp for schedule(static)       // versione 3
```

### 7.1 Che cos'è lo scheduling

Lo **schedule** decide quali iterazioni vanno a quale thread. Con `static` e senza numero, le iterazioni sono divise in blocchi contigui, uno per thread. Con `static,1` (*cyclic*, ciclico) si assegnano a rotazione, una iterazione per volta.

Esempio: righe `i = 1..8`, 4 thread.

```
 schedule(static)      T0: 1 2    T1: 3 4    T2: 5 6    T3: 7 8
 schedule(static,1)    T0: 1 5    T1: 2 6    T2: 3 7    T3: 4 8
```

### 7.2 Perché `static,1` potrebbe aiutare

Al passo `k` si lavora sulle righe `k+1 .. n-1`. Al passo successivo sulle righe `k+2 .. n-1`: l'intervallo si accorcia.

- Con **`static` a blocchi**, i confini dei blocchi si spostano a ogni passo, quindi una riga può passare da un thread all'altro. Quando una riga cambia thread, la sua copia nella cache del vecchio core è inutile: il nuovo core deve ricaricarla dalla memoria, che è lenta.
- Con **`static,1`**, la riga `i` va sempre al thread `i mod T` (con `T` thread), qualunque sia `k`. Ogni thread rilavora sulle **stesse righe** passo dopo passo, quindi le trova ancora nella propria cache.

In più il bilanciamento resta buono anche quando restano poche righe.

### 7.3 Perché potrebbe invece peggiorare

La distribuzione ciclica ha un lato negativo: i thread lavorano su righe **non contigue** e alternate. Righe adiacenti, scritte da thread diversi, sono vicine in memoria e possono condividere linee di cache ai bordi (*false sharing*). Il vantaggio dipende da quanto la matrice sta in cache e da quanti core ci sono.

### 7.4 Che cosa dicono le misure

Nel test fatto durante la preparazione (2 core, `n = 1200`), la versione 3 (`static`) è stata **più veloce** della 4 (`static,1`) di circa il 27% con 2 thread. Con 2 core e una matrice piccola l'effetto cache non emerge, quindi non è una smentita dell'idea teorica. Il confronto va rifatto sul tuo hardware con più core e `n` grande, per esempio `./bench.sh 4000 7 "1 2 4 8 16"`: `bench.sh` stampa una tabella dedicata con il rapporto `T(static) / T(static,1)` per ogni numero di thread. Sopra 1 vince `static,1`, entro ±3% è rumore.

Morale: un'ottimizzazione ben argomentata in teoria **va comunque misurata**. Se non migliora, si scarta.

---

## 8. Come compilare, eseguire e leggere i risultati

### Compilare ed eseguire

```bash
make                    # compila lu_v0 ... lu_v4
make check              # verifica la correttezza di tutte le versioni
make bench N=4000 REPS=7 THREADS="1 2 4 8"
OMP_NUM_THREADS=4 ./lu_v3 2000 5 1   # una versione: N, ripetizioni, controllo
```

### Verificare la correttezza

`make check` confronta ogni versione con la baseline seriale (`diff`) e calcola il **residuo** `max|L·U − A| / max|A|`. Valori intorno a `1e-13` / `1e-15` sono ottimi. Le versioni 1-2 danno differenza zero, le 3-4 circa `3e-13` (per reciproco e FMA, vedi 6.1).

### Metriche

- **Speedup:** `S(p) = T_seriale / T_parallelo(p)`. Con 4 thread, un valore vicino a 4 è ideale.
- **Efficienza:** `E(p) = S(p) / p`. Se scende molto sotto `0,7` c'è un collo di bottiglia (sincronizzazione o banda di memoria).
- **GFLOP/s:** `(2/3)·n³ / tempo / 10⁹`. Misura quanto lavoro utile si fa per secondo.

### Risultati di esempio (indicativi, 2 core, `n = 1200`)

| Versione | Speedup con 2 thread |
|---|---|
| 1: `parallel for` | 2,05 |
| 2: regione unica | 1,94 |
| 3: micro-ottimizzazioni | 2,09 |
| 4: `static,1` | 1,51 |

Con soli 2 core le differenze tra le versioni sono piccole. Non si possono trarre conclusioni generali: servono più core e `n` più grande.

---

## 9. Errori tipici e glossario

### Errori tipici

1. **Data race:** due thread toccano la stessa cella senza sincronizzazione. Risultati sbagliati a volte sì, a volte no.
2. **Variabile condivisa invece di privata:** tipicamente un indice di ciclo interno senza `private`.
3. **`nowait` messo dove serviva la barriera:** vedi 5.6.
4. **Misurare con `n` troppo piccolo:** si misura solo l'overhead di OpenMP, non il calcolo.
5. **Confrontare con la versione OpenMP a 1 thread** invece che con la vera baseline seriale: nasconde l'overhead.

### Glossario

| Termine | Significato |
|---|---|
| Thread | Filo di esecuzione. Più thread nello stesso programma condividono la memoria. |
| Team | L'insieme dei thread che eseguono una regione parallela. |
| Fork / join | Creazione (o risveglio) del team all'inizio di una regione e riunione alla fine. |
| Regione parallela | Il blocco di codice dopo `#pragma omp parallel`, eseguito da tutti i thread del team. |
| Worksharing | Divisione del lavoro tra i thread (per esempio `omp for`). |
| Barriera | Punto in cui tutti i thread si aspettano prima di proseguire. |
| Barriera implicita | Barriera inserita automaticamente alla fine di `omp for` (eliminabile con `nowait`). |
| `shared` / `private` | Variabile con una sola copia per tutti / una copia per ogni thread. |
| Data race | Accesso concorrente non sincronizzato alla stessa cella, con almeno una scrittura. |
| Schedule | Regola con cui le iterazioni di un ciclo sono assegnate ai thread. |
| SIMD | Una istruzione che opera su più dati insieme (registri vettoriali). |
| FMA | Moltiplicazione e somma in una sola istruzione con un solo arrotondamento. |
| Aliasing | Due puntatori che possono riferirsi alla stessa memoria. `restrict` promette che non succede. |
| Cache | Memoria piccola e veloce vicina al core. Riusare dati già in cache accelera molto. |
| False sharing | Thread diversi scrivono celle diverse che stanno nella stessa linea di cache, che rimbalza tra i core. |
| Speedup / efficienza | Quante volte più veloce rispetto al seriale / speedup diviso il numero di thread. |
