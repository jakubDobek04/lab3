# Lab 3 — Collaborative Filtering (rekomendacje według ocen)

## Co robi moduł
Silnik rekomendacji user-based CF. Z `ratings.csv` (12 użytkowników × 24 filmy, 182 oceny)
buduje macierz gustów, centruje oceny na średnią każdego użytkownika, liczy cosine
podobieństwa między ludźmi i przewiduje brakujące oceny ważoną średnią odchyleń sąsiadów:

    predicted(u, m) = mean(u) + Σ sim(u,v)·(r(v,m) − mean(v)) / Σ|sim(u,v)|

Parametry: `top_n` (ile poleceń), `k` (ilu sąsiadów głosuje), `min_common`
(min. wspólnych ocen, by uznać kogoś za sąsiada).

## Jak uruchomić
    py recommender.py                    # szybki test silnika (bez serwera)
    py app.py                            # Flask na http://127.0.0.1:5000
    py -m pytest test_lab03.py -q        # testy (16 passed)

## Endpointy
- `GET /api/recommend?user=Ania[&top_n=5][&k=3][&min_common=2]`
- `GET /api/users`
- `GET /api/similar?user=Ania`

Błędy: 400 — brak/zły parametr; 404 — nieznany user (+ lista `known_users`).

## Przykładowa odpowiedź
`GET /api/recommend?user=Ania&top_n=3`

```json
{
  "count": 3,
  "recommendations": [
    {
      "because": [
        "Celina",
        "Jacek",
        "Kasia"
      ],
      "predicted": 4.31,
      "title": "Fight Club"
    },
    {
      "because": [
        "Celina"
      ],
      "predicted": 3.08,
      "title": "Forrest Gump"
    },
    {
      "because": [
        "Celina",
        "Jacek",
        "Kasia"
      ],
      "predicted": 2.37,
      "title": "Toy Story"
    }
  ],
  "user": "Ania"
}