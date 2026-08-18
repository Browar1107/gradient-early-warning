# Wczesne ostrzeganie przed częściowym kolapsem routingu

Orkiestrator uczony RL kieruje zadania do workerów. **Kolaps częściowy** to
sytuacja, w której kilku workerów przestaje dostawać ruch, a zagregowany rozkład
routingu dalej wygląda zdrowo — utrata trzech specjalistów z dwudziestu prawie
nie rusza `max_k pi_k`. Średnia nagroda też nie spada, bo zadania tych
specjalistów są rzadkie.

Pytanie: **czy gradient polityki ostrzega przed tym wcześniej niż sam rozkład?**

Odpowiedź w skrócie: **monitoruj tempo, nie poziom — a gradient dokłada resztę.**

Największy zysk nie pochodzi z gradientu, tylko z patrzenia na **tempo zmiany**
rozkładu zamiast na jego poziom. Gradient dokłada to, czego polityka nie widzi
wcale: kolapsy postępujące pod mitygacją, która utrzymuje rozkład rozproszony.

## Metoda

**Sygnał.** Dla polityki softmax pochodna log-prawdopodobieństwa po logicie to
`1[a=k] − pi_k`, więc

```
g_k = E[ A · (1[a=k] − pi_k) ]
```

to średnia presja gradientu w stronę workera *k* w update'cie, który zaraz
zostanie zastosowany. Czyta się ją **przed** krokiem optymalizatora — to jest
powód, dla którego może wyprzedzać stan.

Do kolapsu częściowego statystyką alarmową jest **`min_k g_k`** — najbardziej
ujemna presja, wskazująca workera spychanego w dół. Agregat `max_k g_k` jest na
to ślepy, tak samo jak `max_k pi_k`.

**Sygnał jest transientem, nie poziomem.** Gdy `pi_k → 0`, worker przestaje być
próbkowany, więc `g_k = E[A(0 − pi_k)] = −pi_k·E[A] → 0`. Presja znika, gdy
worker już umarł. Szuka się dołka poprzedzającego śmierć, nie poziomu po niej.

**Definicja awarii.** Nie próg na rozkładzie routingu. Dla każdego workera
pokrycie `c_k = P(routing do k | zadanie z held-outu, dla którego k jest
właściwym wyborem)`. Worker martwy przy `c_k < 0,5`; system zawiódł, gdy umiera
trzeci. Żaden monitor nie czyta tej wielkości.

To rozróżnienie jest krytyczne. Monitor oceniany względem definicji zapisanej
w jego własnych jednostkach wygrywa z konstrukcji, a jego pozorne wyprzedzenie
to tylko odległość między dwoma progami na jednej wielkości.

**Ewaluacja.** Monitory porównuje się wyłącznie **przy dopasowanym
false-alarm rate**. Bieg null musi być takim, w którym sygnał się rusza, a
awaria nie następuje — bieg, w którym nic się nie dzieje, daje każdemu
monitorowi darmowe FAR = 0 i nic nie mierzy.

## Wynik

Przy FAR ≤ 10%, 13 biegów z awarią i 11 null:

| monitor | detekcja | lead | kolaps szybki | wolny pod mitygacją |
|---|---|---|---|---|
| `max_k pi_k` agregat | 0% | — | 0/6 | 0/6 |
| `min_k pi_k` poziom | 31% | 35 | 4/6 | 0/6 |
| **`min_k pi_k` tempo** | **69%** | **153** | **6/6** | 3/6 |
| `min_k g_k` presja | 54% | 178 | 1/6 | **6/6** |

**Tempo + presja razem: 92% detekcji przy 18% fałszywych alarmów** — więcej niż
którakolwiek para bez gradientu. Presja jest jedynym monitorem łapiącym
wszystkie kolapsy pod mitygacją.

Ranking zależy jednak od tego, jak trudna jest pula null. Powyżej ponad połowę
stanowią biegi, w których w momencie przesunięcia nie dzieje się nic. Gdy zostawić
wyłącznie biegi z prawdziwym, ale nieszkodliwym przesunięciem, **tempo i presja
remisują na detekcji (69%), a presja daje o połowę dłuższe wyprzedzenie**
(237 wobec 153 kroków). Szczegóły w `METODY.md`.

Hipotezy zapisano przed uruchomieniem — `PREREGISTRATION.md`. Dwie z pięciu
okazały się błędne. Monitor tempa **nie był prerejestrowany**: dodano go po
zobaczeniu danych i jest tam oznaczony jako analiza po fakcie.

## Uruchomienie

```bash
pip install -r requirements.txt
python toy_partial_collapse.py    # -> data/partial_*.npz   (~8 min)
python example.py                 # minimalny przykład użycia metody
```

Notebook `partial_collapse.ipynb` jest zapisany z outputami — można go czytać
bez uruchamiania.

| plik | co to jest |
|---|---|
| `detectors.py` | **metoda.** Czysty numpy, bez torcha. To się przenosi. |
| `toy_partial_collapse.py` | system syntetyczny: 20 workerów, przesunięcie ruchu |
| `partial_collapse.ipynb` | analiza z wykresami |
| `example.py` | minimalny przykład end-to-end na sztucznych tablicach |

## System syntetyczny

Dwudziestu workerów: jeden generalista i dziewiętnastu specjalistów, każdy
wyłącznie właściwy dla własnej niszy zadań. Nagroda binarna — punkt, jeśli
wybrany worker odniósł sukces.

Faza 1: wszystkie nisze równie częste, każdy specjalista uczy się swojej.
Od kroku `t_shift` rozkład ruchu się zmienia i sześć nisz staje się rzadkich.
Ci specjaliści przestają pojawiać się w gradiencie, parametry, na których się
opierali, zostają nadpisane przez ruch, który został, i ich routing zanika.

Warunki: `shift` (przesunięcie ostre), `mild_shift` (łagodne), `stable` (brak),
`entropy_reg` (przesunięcie plus podniesiony bonus entropijny od `t_shift` —
mitygacja, którą realnie by się wdrożyło). Etykieta pozytyw/null wynika z tego,
co się w biegu stało, nie z warunku, z którego pochodzi.

## Przeniesienie na prawdziwy system

Loguj na krok: rozkład `pi` po workerach, wylosowaną akcję, advantage.
Osobno, rzadziej: pokrycie na stałym zbiorze held-out.

```python
from detectors import pressure, ema, coverage_failure, sweep, table_at_far

g_k = pressure(advantage, action, pi, per_worker=True)   # (K,) na krok
stat_g = -ema([g.min() for g in g_history])              # presja
pmin = ema(pi_history.min(axis=1))
stat_slope = -ema(np.diff(pmin, prepend=pmin[0]), alpha=0.05)   # tempo

t_fail, alive = coverage_failure(coverage, eval_steps, start_idx=onset)
curves = sweep(positives, nulls, start=onset)
table_at_far(curves, far_budget=0.10)
```

Trzy rzeczy, których nie należy obchodzić:

- **Definiuj awarię przez to, co tracisz** — trafność na zadaniach obsługiwanych
  wyłącznie przez konkretnych workerów — a nie progiem na rozkładzie routingu.
- **Monitoruj tempo, nie poziom.** Pochodna `min_k pi_k` bije próg na samym
  `min_k pi_k` (69% wobec 31%) i nie wymaga dostępu do gradientu.
- **Używaj presji per-worker.** Agregat nie widzi kolapsu częściowego, a presja
  łapie kolapsy pod mitygacją, których tempo nie łapie. Wdrażaj oba naraz.
- **Sprawdź, czy działa u Ciebie mitygacja koncentracji.** Jeśli tak, monitor
  rozkładu może być ślepy na kolaps, który postępuje — i to jest dokładnie
  reżim, w którym ta metoda ma przewagę.

## Ograniczenia

Detekcja jest niska bezwzględnie: najlepszy pojedynczy monitor łapie 54% awarii
przy FAR 10%. To dowód, że sygnał niesie informację niedostępną monitorom
rozkładu, a nie gotowy system produkcyjny.

Główne tabele liczą alarmy od momentu przesunięcia, co daje monitorom darmową
informację o onsecie. Test odporności bez niej (alarmy od kroku 600, 400 kroków
przed zdarzeniem): tempo bez zmian 69%, **presja spada 54% → 38%**, para
85% przy FAR 9%. Liczby dla presji w tabelach są zawyżone; para się broni.
Na prawdziwym systemie próg trzeba kalibrować na okresie stabilnej pracy.

Dwadzieścia workerów i jedno źródło kolapsu. Nie wiadomo, czy przewaga presji
utrzyma się przy innych mechanizmach zaniku routingu.
