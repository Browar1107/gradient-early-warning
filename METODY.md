# Opis metody — wykrywanie częściowego kolapsu routingu

Opisuje **co metoda robi**, **jaki wynik daje na systemie syntetycznym** i **jak
przenieść test na prawdziwy router LLM**.

Kod do portu: `detectors.py` (czysty numpy, bez zależności od frameworka
treningowego). Hipotezy i wyniki: `PREREGISTRATION.md`.

---

## Problem

Orkiestrator uczony RL kieruje zadania do workerów. **Kolaps częściowy** to
utrata kilku workerów przy zachowaniu zdrowo wyglądającego agregatu.

Dlaczego jest trudny do zauważenia:

- `max_k pi_k` prawie się nie rusza — trzech workerów z dwudziestu to niewielka
  masa, rozproszona na wiele zadań;
- średnia nagroda nie spada, bo zadania utraconych specjalistów są rzadkie;
- entropia polityki spada nieznacznie i nieodróżnialnie od normalnej zbieżności.

Traci się natomiast dokładnie to, po co był router: pokrycie zadań, które umie
obsłużyć wyłącznie jeden konkretny worker.

---

## Metoda

### 1. Sygnał: presja gradientowa per worker

Dla polityki softmax pochodna log-prawdopodobieństwa po logicie wynosi
`1[a=k] − pi_k`, więc

```
g_k = E[ A · (1[a=k] − pi_k) ]
```

to średnia presja gradientu w stronę workera *k* w update'cie, który zaraz
zostanie zastosowany. Odczytuje się ją **przed** krokiem optymalizatora — stąd
możliwość wyprzedzenia stanu.

```python
from detectors import pressure
g_k = pressure(advantage, action, pi, per_worker=True)   # (K,)
```

**Statystyka alarmowa to `min_k g_k`** — najbardziej ujemna presja, wskazująca
workera spychanego w dół. Agregat `max_k g_k` jest ślepy na kolaps częściowy
dokładnie tak samo jak `max_k pi_k`.

### 2. Sygnał jest transientem, nie poziomem

Gdy `pi_k → 0`, worker przestaje być próbkowany, więc

```
g_k = E[A·(0 − pi_k)] = −pi_k·E[A] → 0
```

Presja **znika**, gdy worker już umarł. Zmierzony przebieg: +0,0004 przed
przesunięciem → dołek −0,0050 → +0,0003 na końcu, przy `pi_k` spadającym
z 0,037 do 0,005.

Konsekwencja praktyczna: alarm oparty na progu, który ma się utrzymać po
śmierci workera, nie zadziała. Szuka się dołka poprzedzającego śmierć.

### 3. Definicja awarii — na held-oucie, nie na rozkładzie

Dla każdego workera pokrycie

```
c_k = P(routing do k | zadanie z held-outu, dla którego k jest właściwym wyborem)
```

Worker martwy przy `c_k < 0,5`; system zawiódł, gdy umiera trzeci.

```python
from detectors import coverage_failure
t_fail, alive = coverage_failure(coverage, eval_steps, start_idx=onset)
```

**To jest najważniejsza decyzja projektowa w całym teście.** Monitor oceniany
względem definicji zapisanej w jego własnych jednostkach wygrywa z konstrukcji,
a jego pozorne wyprzedzenie to tylko odległość między dwoma progami na jednej
wielkości. Definicja oparta na pokryciu nie jest czytana przez żaden monitor.

### 4. Ewaluacja — wyłącznie przy dopasowanym FAR

```python
from detectors import sweep, table_at_far
curves = sweep(positives, nulls, start=onset)
table_at_far(curves, far_budget=0.10)
```

Dwie reguły:

- **Nigdy nie porównuj monitorów przy różnych false-alarm rate.** Monitor
  alarmujący bez przerwy wygrywa na wyprzedzeniu i jest bezwartościowy.
- **Bieg null musi być takim, w którym sygnał się rusza, a awaria nie
  następuje.** Bieg, w którym nic się nie dzieje, daje każdemu monitorowi
  darmowe FAR = 0 i nie mierzy niczego.

---

## Wynik na systemie syntetycznym

Dwudziestu workerów, 6 seedów × 4 warunki, 13 biegów z awarią i 11 null.

| monitor | FAR ≤ 0% | FAR ≤ 10% | FAR ≤ 20% |
|---|---|---|---|
| `pi_max` (agregat) | niewykonalne | niewykonalne | DR 8%, lead 462 |
| `pi_min` (poziom) | DR 31%, lead 30 | DR 31%, lead 35 | DR 46%, lead 34 |
| **`pi_slope` (tempo)** | **DR 62%, lead 133** | **DR 69%, lead 153** | **DR 77%, lead 169** |
| `neg_entropy` | niewykonalne | niewykonalne | niewykonalne |
| **`g_min` (presja)** | **DR 46%, lead 177** | **DR 54%, lead 178** | **DR 69%, lead 235** |
| `grad_norm` | DR 8%, lead 243 | DR 8%, lead 245 | DR 8%, lead 257 |

### Kiedy co działa

| | `pi_min` poziom | `pi_slope` tempo | `g_min` presja |
|---|---|---|---|
| kolaps szybki (~80 kroków po onsecie) | 4/6, lead 8–56 | **6/6, lead 58–105** | 1/6, lead 16 |
| kolaps wolny pod mitygacją (~270 kroków) | 0/6 | 3/6, lead 235–314 | **6/6, lead 142–290** |

### Pula null decyduje o rankingu

Pula null powyżej to 6 biegów `stable` (w kroku przesunięcia nie dzieje się nic)
i 5 `mild_shift` (przesunięcie nastąpiło, nikogo nie zabiło). Ponad połowa nie
stawia żadnego pytania. Wersja twardsza — null wyłącznie `mild_shift`:

| monitor | null = wszystkie (n=11) | null = tylko `mild_shift` (n=5) |
|---|---|---|
| `pi_min` poziom | DR 46%, lead 34 | DR 31%, lead 35 |
| `pi_slope` tempo | **DR 77%**, lead 169 | DR 69%, lead 153 |
| `g_min` presja | DR 69%, lead 235 | DR 69%, lead **237** |

**Na twardej puli tempo i presja remisują na detekcji, a presja daje o połowę
dłuższe wyprzedzenie.** Przewaga tempa znika razem z darmowymi biegami `stable`.

Czasy alarmów tłumaczą mechanizm: tempo strzela 2–15 kroków po przesunięciu —
reaguje na wielkość skoku. Presja strzela 17–178 kroków po nim — śledzi proces.
Na łatwej puli wygrywa refleks, na twardej zdolność odróżnienia zmiany groźnej
od nieszkodliwej. (n=5, najmniejszy niezerowy FAR to 20% — do rozstrzygnięcia
potrzeba więcej seedów.)

**Największy pojedynczy zysk daje przejście z poziomu na tempo** — bez żadnego
gradientu, samą pochodną rozkładu: 31% → 69% detekcji.

**Gradient dokłada pokrycie, którego polityka nie daje.** Presja jest jedynym
monitorem łapiącym wszystkie 6 kolapsów pod mitygacją. Para tempo + presja daje
**DR 92% przy FAR 18%**, wobec 77% dla poziom + presja i 69% dla tempo + poziom.

Mechanizm: bonus entropijny podniesiony w momencie przesunięcia utrzymuje
rozkład rozproszony, więc `pi_min` nie spada i monitor rozkładu jest ślepy.
Presja `g_k` mimo to schodzi na minus, bo struktura nagrody dalej spycha tych
workerów.

> **Mitygacja usuwa objaw widoczny w poziomie rozkładu, nie usuwając przyczyny
> widocznej w gradiencie.**

Pod mitygacją poziom `min_k pi_k` nie tylko nie spada — on rośnie. Tempo łapie
połowę takich biegów, presja wszystkie. To reżim realistyczny, bo bonus
entropijny, KL do priora i kary za koncentrację to standardowe mitygacje.

**Uwaga metodologiczna:** monitor tempa nie był prerejestrowany — dodano go po
obejrzeniu danych, gdy okazało się, że kształt krzywej jest wzrokowo oczywisty,
mimo że żaden próg na poziomie go nie łapie. Szczegóły w `PREREGISTRATION.md`.
Wynik post-hoc wymaga replikacji na niezależnych seedach.

---

## Przeniesienie na prawdziwy router LLM

### Co logować

| co | kiedy | kształt |
|---|---|---|
| `pi` — rozkład po workerach | każdy krok treningu | `(batch, K)` |
| `action` — wylosowany worker | każdy krok | `(batch,)` |
| `advantage` — nagroda minus baseline | każdy krok | `(batch,)` |
| `coverage` — pokrycie na held-oucie | co N kroków | `(K−1,)` |

Trzy pierwsze i tak są w pętli treningowej. Koszt dodatkowy to jeden forward
pass na zbiorze ewaluacyjnym co N kroków.

### Jak zbudować zbiór held-out

Potrzebny jest zestaw zadań, dla których wiadomo, który worker jest właściwym
wyborem — w praktyce zadania, na których dokładnie jeden model lub narzędzie ma
istotnie wyższy wynik niż reszta. Nie musi być duży; kilkadziesiąt przykładów
na workera wystarcza do stabilnego `c_k`.

Jeśli nie da się przypisać właściwego workera, użyj wprost trafności na wycinku:
`c_k` = jakość na zadaniach typu *k*, znormalizowana do wartości z okresu
stabilnego. Definicja awarii pozostaje niezależna od rozkładu.

### Pętla

```python
from detectors import pressure, ema, coverage_failure, sweep, table_at_far

g_history.append(pressure(advantage, action, pi, per_worker=True))
stat = -ema([g.min() for g in g_history])          # statystyka alarmowa

t_fail, alive = coverage_failure(coverage, eval_steps, start_idx=onset)
curves = sweep(positives, nulls, start=onset)
op = table_at_far(curves, far_budget=0.10)
```

`op["thr"]` to próg skalibrowany przy zadanym budżecie fałszywych alarmów —
i to jedyny sensowny sposób jego doboru.

### Czego się spodziewać, a czego nie

**Spodziewaj się**, że agregat rozkładu nie wykryje niczego. To najlepiej
potwierdzona część wyniku.

**Spodziewaj się przewagi presji tylko wtedy**, gdy w systemie działa mitygacja
koncentracji. Bez niej rozkład porusza się szybciej, niż wygładzony sygnał
gradientowy zdąży przekroczyć próg, i to rozkład wygrywa.

**Nie spodziewaj się wysokiej detekcji.** Najlepszy pojedynczy monitor łapie
54% awarii przy FAR 10%. Do wdrożenia potrzebne są oba monitory naraz.

**Onset — zmierzone, nie oszacowane.** Główne tabele liczą alarmy od kroku
przesunięcia, co daje każdemu monitorowi darmową informację o momencie
zdarzenia. Jest ona warta sporo: przed przesunięciem presja spędza ok. 25% czasu
pod progiem, a przy liczeniu alarmów od kroku 0 **oba monitory alarmują na
11/11 biegów null**.

Wariant bez tego prezentu — alarmy liczone od kroku 600, czyli po rozruchu, ale
400 kroków przed przesunięciem:

| monitor | onset znany | onset nieznany |
|---|---|---|
| `pi_min` poziom | DR 31%, lead 35 | DR 31%, lead 34 |
| `pi_slope` tempo | DR 69%, lead 153 | **DR 69%, lead 153** |
| `g_min` presja | DR 54%, lead 178 | **DR 38%, lead 163** |
| para tempo + presja | DR 92% przy FAR 18% | **DR 85% przy FAR 9%** |

Tempo jest odporne. **Presja traci jedną trzecią detekcji** — jej pojedyncze
liczby z głównych tabel są zawyżone. Para broni się (85%), a presja dalej łapie
5/6 kolapsów pod mitygacją.

Praktyczny wniosek: **próg kalibruj na okresie stabilnej pracy**, nie na oknie
wokół podejrzewanego zdarzenia, i licz się z niższą detekcją niż w tabelach.

---

## Ograniczenia

- Dwudziestu workerów i jedno źródło kolapsu (przesunięcie rozkładu ruchu).
  Nie wiadomo, czy przewaga presji utrzyma się przy innych mechanizmach zaniku.
- Router to mały MLP. W prawdziwym systemie orkiestrator sam bywa modelem
  językowym i `pi` trzeba wyciągać z rozkładu po tokenach wyboru narzędzia.
- Nagroda binarna. Przy nagrodzie z modelu nagradzającego `advantage` niesie
  własny szum, którego ten test nie modeluje.
