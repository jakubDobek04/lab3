from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flask import Flask, jsonify, request, send_from_directory

if __package__:
    from .recommender import matrix, recommend, similar_users
else:
    from recommender import matrix, recommend, similar_users

app = Flask(__name__, static_folder="../assets", static_url_path="/assets")


@app.get("/")
def home():
    return send_from_directory(app.root_path, "index.html")


@app.route("/api/recommend")
def api_recommend():
    user = request.args.get("user", "").strip()
    if not user:
        return jsonify({"error": "podaj parametr ?user=..."}), 400

    count = request.args.get("top_n", type=int) if "top_n" in request.args else 5
    if "top_n" in request.args and (count is None or not 1 <= count <= 10):
        return jsonify({"error": "top_n musi być od 1 do 10"}), 400

    k = request.args.get("k", type=int) if "k" in request.args else 3
    if "k" in request.args and (k is None or not 1 <= k <= 11):
        return jsonify({"error": "k musi być od 1 do 11"}), 400

    min_common = request.args.get("min_common", type=int) if "min_common" in request.args else 2
    if "min_common" in request.args and (min_common is None or min_common < 1):
        return jsonify({"error": "min_common musi być >= 1"}), 400

    recs = recommend(user, top_n=count, k=k, min_common=min_common)
    if recs is None:
        return jsonify({"error": "nie znam takiego uzytkownika",
                        "known_users": sorted(matrix.index.tolist())}), 404
    return jsonify({"user": user, "count": len(recs), "recommendations": recs})


@app.route("/api/users")
def api_users():
    return jsonify({"count": len(matrix.index), "users": sorted(matrix.index.tolist())})


@app.route("/api/similar")
def api_similar():
    user = request.args.get("user", "").strip()
    if not user:
        return jsonify({"error": "podaj parametr ?user=..."}), 400
    s = similar_users(user)
    if s is None:
        return jsonify({"error": "nie znam takiego uzytkownika"}), 404
    return jsonify({"user": user, "similar": s})


if __name__ == "__main__":
    app.run(debug=True)