# Prerejestracja

Sekcję **Hipotezy** zapisano przed uruchomieniem eksperymentu. Sekcję
**Wyniki** dopisano po. Hipotez nie zmieniano.

## Hipotezy

Oznaczenia: `pi_max` = największe prawdopodobieństwo routingu, `pi_min` =
najmniejsze, `g_min` = najbardziej ujemna presja gradientowa `min_k g_k`.
Awaria zdefiniowana pokryciem na held-oucie (opis w `README.md`).

**H1 — agregat jest ślepy.** `pi_max` wykrywa mniej niż 30% awarii przy FAR
poniżej 10%. Utrata trzech workerów z dwudziestu nie rusza maksimum.
*Pewność: 0,9*

**H2 — `pi_min` jest silnym baselinem.** Wykrywa ≥ 80% przy FAR ≤ 10%, lead
20–60 kroków. *Pewność: 0,75*

**H3 — `g_min` wyprzedza `pi_min`.** Dla polityki softmax `g_k` jest
proporcjonalne do `d log(pi_k)/dt`, a poziom przekracza próg po tempie.
Przewidywany lead 40–120 kroków. *Pewność, że lead istnieje: 0,6*

**H4 — test rozstrzygający; spodziewam się, że nie przejdzie.** Hipoteza
potwierdzona tylko jeśli `g_min` osiąga lead ≥ 1,5× lead `pi_min` przy
FAR ≤ 10%, na ≥ 5 z 6 seedów. *Pewność: 0,35*

Powód: `g_k` to estymator z batcha o wariancji ~`1/sqrt(pi_k · B)` — najgorszy
dokładnie dla workerów już malejących. Wygładzanie potrzebne do kontroli FAR
kosztuje lead.

**H5 — sygnał jest transientem.** `g_k` umierającego workera schodzi na minus,
osiąga dołek i wraca do zera: przy `pi_k → 0` worker nie jest próbkowany, więc
`g_k = −pi_k·E[A] → 0`. Szuka się dołka, nie poziomu. *Pewność: 0,85*

---

## Wyniki

13 pozytywów, 11 biegów null, 6 seedów × 4 warunki.

| monitor | FAR ≤ 0% | FAR ≤ 10% | FAR ≤ 20% |
|---|---|---|---|
| `pi_max` | niewykonalne | niewykonalne | DR 8%, lead 462 |
| `pi_min` | DR 31%, lead 30 | DR 31%, lead 35 | DR 46%, lead 34 |
| `neg_entropy` | niewykonalne | niewykonalne | niewykonalne |
| **`g_min`** | **DR 46%, lead 177** | **DR 54%, lead 178** | **DR 69%, lead 235** |
| `grad_norm` | DR 8%, lead 243 | DR 8%, lead 245 | DR 8%, lead 257 |

| | wynik |
|---|---|
| **H1** | ✅ `pi_max` nie wykrywa nic przy FAR ≤ 10% |
| **H2** | ❌ DR 31%, nie ≥ 80%. Lead trafiony (35 wobec 20–60), skuteczność mocno przeszacowana |
| **H3** | ✅ co do kierunku, lead 178 wobec przewidzianych 40–120 |
| **H4** | ❌ nie przeszedł jak zapisany — patrz niżej |
| **H5** | ✅ dokładnie: +0,00040 → dołek −0,00497 → +0,00028, przy `pi_k` 0,037 → 0,005 |

### H4: próg leadu przekroczony, kryterium seedów nie

Zbiorczo `g_min` daje lead 5,1× większy niż `pi_min` przy FAR ≤ 10% — próg 1,5×
z zapasem. Ale rozbicie na warunki pokazuje, że uśrednianie ukrywało tu wszystko:

| | wykrywa `pi_min` | wykrywa `g_min` |
|---|---|---|
| kolaps szybki (`shift`, ~80 kroków po onsecie) | **4/6**, lead 8–56 | 1/6 |
| kolaps wolny pod mitygacją (`entropy_reg`, ~270 kroków) | **0/6** | **6/6**, lead 142–290 |

**Monitory są komplementarne, nie konkurencyjne.** Na 13 pozytywach: 3 wykrywa
tylko `pi_min`, 6 tylko `g_min`, 1 oba. Suma: DR 77% przy FAR 18%.

Mechanizm: bonus entropijny podniesiony w momencie przesunięcia utrzymuje
rozkład rozproszony, więc `pi_min` nie spada i monitor rozkładu jest ślepy.
Presja `g_k` mimo to schodzi na minus, bo struktura nagrody dalej spycha tych
workerów. **Gdy mitygacja trzyma stan w miejscu, tylko presja pokazuje, że
problem trwa.** Tego nie przewidywała żadna z pięciu hipotez.

---

## Analiza po fakcie (NIE prerejestrowana)

Zestaw monitorów w hipotezach zawierał wyłącznie progi na **poziomie**
statystyki. Nie zawierał żadnego monitora **tempa zmiany**. To była wada
projektu, nie wynik — zauważona po obejrzeniu wykresów, gdy okazało się, że
kształt `min_k pi_k` przy przesunięciu jest wzrokowo oczywisty, mimo że żaden
progowy monitor go nie łapie.

Monitor dodany po fakcie: `pi_slope = -EMA(Δ min_k pi_k)`, różnica wsteczna
(wyłącznie przeszłość, bez podglądania kroku w przód).

| monitor | FAR ≤ 0% | FAR ≤ 10% | FAR ≤ 20% |
|---|---|---|---|
| `pi_min` poziom | DR 31%, lead 30 | DR 31%, lead 35 | DR 46%, lead 34 |
| **`pi_slope` tempo** | **DR 62%, lead 133** | **DR 69%, lead 153** | **DR 77%, lead 169** |
| `g_min` presja | DR 46%, lead 177 | DR 54%, lead 178 | DR 69%, lead 235 |

| | `pi_min` poziom | `pi_slope` tempo | `g_min` presja |
|---|---|---|---|
| kolaps szybki | 4/6, lead 8–56 | **6/6, lead 58–105** | 1/6, lead 16 |
| kolaps wolny pod mitygacją | 0/6 | 3/6, lead 235–314 | **6/6, lead 142–290** |

**Konsekwencja: H4 była testowana przeciwko za słabemu baselinowi.** Presja
wygrywała z progiem na poziomie, ale monitor tempa — trywialny i liczony
wyłącznie z polityki — bije ją na detekcji (69% wobec 54%) i łapie połowę
kolapsów pod mitygacją.

Teza „gdy mitygacja trzyma stan w miejscu, tylko presja pokazuje problem" jest
**za mocna**. Stan nie stoi; ma charakterystyczny kształt, którego próg na
poziomie nie widzi, a pochodna widzi.

Co zostaje w mocy:

- agregaty (`pi_max`, entropia) są ślepe — bez zmian;
- presja jest jedynym monitorem łapiącym **wszystkie** 6 kolapsów pod mitygacją;
- najlepsza para to tempo + presja: **DR 92% przy FAR 18%**, wobec 77% dla
  poziom + presja i 69% dla tempo + poziom. Gradient wnosi pokrycie, którego
  sama polityka nie daje — mniej, niż wskazywał pierwotny odczyt.

Ten wynik jest post-hoc i jako taki wymaga replikacji na niezależnych seedach,
zanim zostanie potraktowany na równi z prerejestrowanymi.

---

### Czego to nie pokazuje

Detekcja jest niska bezwzględnie — najlepszy pojedynczy monitor łapie 54%
awarii przy FAR 10%. To dowód, że sygnał niesie informację niedostępną
monitorom rozkładu, a nie gotowy system.

Okno ewaluacji startuje w momencie przesunięcia, więc każdy monitor dostaje
onset za darmo. Na prawdziwym systemie onset nie jest znany i leady byłyby
krótsze.
