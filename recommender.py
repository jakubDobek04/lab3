"""LAB 3. silnik rekomendacji v3: collaborative filtering (user-based).

Lab 2 pytało: "jakie filmy są podobne do TEGO filmu?" (treść opisów).
Lab 3 pyta: "kto myśli jak JA, i co ta osoba lubiła?" (oceny ludzi).

Pipeline:
    ratings.csv  ->  pivot (użytkownicy x filmy)  ->  centrowanie na średnią usera
    ->  cosine między użytkownikami na wyśrodkowanych wektorach z zerami za braki
    ->  przewidywanie: średnia usera + średnia ważona odchyleń sąsiadów

Dlaczego centrowanie? Porównujemy odchylenia od własnej średniej, a nie same
dodatnie oceny. Wspólne zera nie zwiększają iloczynu skalarnego. Licznik cosine
uwzględnia wspólnie ocenione filmy, ale normy całe wektory. Przy brakach nie
jest to Pearson liczony wyłącznie na przecięciu ocen; skale i małe przecięcie
nadal wymagają ostrożnej interpretacji.
"""
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv(Path(__file__).resolve().with_name("ratings.csv"))

# macierz użytkownik x film; NaN = brak oceny, nie informacja o oglądaniu
matrix = df.pivot_table(index="user", columns="title", values="rating")

# odchylenie od własnej średniej; braki = 0 (neutralne po centrowaniu)
means = matrix.mean(axis=1)
centered = matrix.sub(means, axis=0).fillna(0)

sim = pd.DataFrame(
    cosine_similarity(centered),
    index=matrix.index,
    columns=matrix.index,
)


def _common_count(user, other):
    """Ile filmów user i other ocenili OBOJE (nie-NaN w obu wierszach)."""
    both = matrix.loc[[user, other]].notna().all(axis=0)
    return int(both.sum())


def recommend(user, top_n=5, k=3, min_common=2):
    """Filmy bez oceny użytkownika, z przewidywaną oceną (1-5).

    predicted(u, m) = mean(u) + sum( sim(u,v) * (r(v,m) - mean(v)) ) / sum(|sim|)
    czyli: startujemy od średniej usera i korygujemy ją tym, jak sąsiedzi
    ocenili film WZGLĘDEM swoich średnich.

    min_common: minimalna liczba wspólnie ocenionych filmów, żeby uznać kogoś
    za sąsiada. Dwa wspólne filmy to bardzo mało — strona lab3 wprost o tym mówi.

    Pole "because" = sąsiedzi, którzy ocenili film i mają niezerową wagę.
    UWAGA: samo nazwisko nie dowodzi, że sąsiad lubi film — jego odchylenie
    i znak podobieństwa mogą podnosić albo obniżać wynik.
    """
    if type(top_n) is not int or top_n < 1 or type(k) is not int or k < 1:
        raise ValueError("top_n i k muszą być dodatnimi liczbami całkowitymi")
    if type(min_common) is not int or min_common < 1:
        raise ValueError("min_common musi być dodatnią liczbą całkowitą")
    if user not in matrix.index:
        return None

    # kandydaci na sąsiadów: posortowani po sim, ale odfiltrowani po min_common
    candidates = sim[user].drop(user).sort_values(ascending=False)
    neighbors = candidates[candidates.index.map(lambda v: _common_count(user, v) >= min_common)].head(k)

    # fallback: jeśli filtr min_common wyciął wszystkich, bierzemy top-k bez filtra
    if neighbors.empty:
        neighbors = candidates.head(k)

    scores = {}

    for movie in matrix.columns:
        if not pd.isna(matrix.loc[user, movie]):
            continue  # user już ocenił. nie polecamy
        num, den = 0.0, 0.0
        for other, w in neighbors.items():
            r = matrix.loc[other, movie]
            if not pd.isna(r):
                num += w * (r - means[other])
                den += abs(w)
        if den > 0:
            pred = means[user] + num / den
            scores[movie] = max(1.0, min(5.0, pred))  # oceny mieszczą się w 1-5

    best = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [
        {
            "title": t,
            "predicted": round(float(s), 2),
            "because": [o for o, weight in neighbors.items()
                        if weight != 0 and not pd.isna(matrix.loc[o, t])],
        }
        for t, s in best
    ]


def similar_users(user, k=3):
    """Kto myśli jak `user`? (do wyzwania)"""
    if type(k) is not int or k < 1:
        raise ValueError("k musi być dodatnią liczbą całkowitą")
    if user not in sim.index:
        return None
    s = sim[user].drop(user).sort_values(ascending=False).head(k)
    return [{"user": u, "similarity": round(float(v), 3)} for u, v in s.items()]


if __name__ == "__main__":
    print('Rekomendacje dla "Ania" (fanka sci-fi):')
    for r in recommend("Ania"):
        print(f'  przewidziane {r["predicted"]}/5  {r["title"]:26s} ocenili: {", ".join(r["because"])}')
    print()
    print("Najbardziej podobni do Ani:", similar_users("Ania"))
    print()
    print("Porównanie k=1 / k=3 / k=10 dla Ani:")
    for kk in (1, 3, 10):
        top = recommend("Ania", top_n=1, k=kk)[0]
        print(f"  k={kk:2d}  ->  {top['title']:26s} {top['predicted']}/5")