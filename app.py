"""
Fraud Detection Web App (Flask + SQLite + scikit-learn)

The transaction dataset is bundled with the app (data/demo_transactions.csv.gz),
so nothing needs to be uploaded.

Run locally:  python app.py
Deploy:       gunicorn app:app
"""
import json
import os
import sqlite3
import datetime as dt

import joblib
import numpy as np
import pandas as pd
from flask import (Flask, Response, flash, g, jsonify, redirect,
                   render_template, request, url_for)

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "model", "fraud_model.joblib")
METRICS_PATH = os.path.join(BASE, "model", "metrics.json")
DATA_PATH = os.path.join(BASE, "data", "demo_transactions.csv.gz")
DB_PATH = os.environ.get("DB_PATH", os.path.join(BASE, "fraud.db"))
SEED_DEMO = os.environ.get("SEED_DEMO", "1") != "0"
TARGET = "Class"
PER_PAGE = 20

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-change-me")

# ---------- model + built-in dataset ----------
for _p in (MODEL_PATH, DATA_PATH):
    if not os.path.exists(_p):
        raise RuntimeError(f"{os.path.relpath(_p, BASE)} not found. Create it with: "
                           "python train.py --train creditcard_fraud.csv")

bundle = joblib.load(MODEL_PATH)
MODEL, THRESHOLD, COLUMNS = bundle["model"], bundle["threshold"], bundle["columns"]

DEMO = pd.read_csv(DATA_PATH)
DEMO_FRAUD = DEMO[DEMO[TARGET] == 1]
DEMO_LEGIT = DEMO[DEMO[TARGET] == 0]


def load_json(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


METRICS = load_json(METRICS_PATH, None)


def score(df):
    probs = MODEL.predict_proba(df[COLUMNS])[:, 1]
    labels = np.where(probs >= THRESHOLD, "FRAUD", "Legit")
    return probs, labels


def sample_demo(n, fraud_rich=True):
    """Random transactions from the built-in dataset (optionally ~20% fraud)."""
    n = min(max(int(n), 1), len(DEMO))
    if fraud_rich:
        n_f = min(max(int(n * 0.2), 1), len(DEMO_FRAUD))
        df = pd.concat([DEMO_FRAUD.sample(n_f), DEMO_LEGIT.sample(n - n_f)])
        return df.sample(frac=1)
    return DEMO.sample(n)


# ---------- database ----------
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(_exc=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def save_rows(db, df, probs, labels, source, actual=None):
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    records = df[COLUMNS].to_dict("records")
    actual = [None] * len(records) if actual is None else [int(a) for a in actual]
    rows = [(now, source, json.dumps({k: float(v) for k, v in r.items()}),
             float(p), str(l), a)
            for r, p, l, a in zip(records, probs, labels, actual)]
    db.executemany(
        "INSERT INTO predictions (created_at, source, features, probability, label, actual) "
        "VALUES (?,?,?,?,?,?)", rows)
    db.commit()


def init_db():
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            source TEXT NOT NULL,
            features TEXT NOT NULL,
            probability REAL NOT NULL,
            label TEXT NOT NULL,
            actual INTEGER
        )""")
    cols = [r[1] for r in con.execute("PRAGMA table_info(predictions)")]
    if "actual" not in cols:  # upgrade older databases
        con.execute("ALTER TABLE predictions ADD COLUMN actual INTEGER")
    con.execute("CREATE INDEX IF NOT EXISTS idx_pred_label ON predictions(label)")
    con.commit()
    # Pre-fill an empty database so the dashboard is never blank on first launch
    if SEED_DEMO and con.execute("SELECT COUNT(*) FROM predictions").fetchone()[0] == 0:
        df = sample_demo(300, True)
        probs, labels = score(df)
        con.row_factory = sqlite3.Row
        save_rows(con, df, probs, labels, "seed", df[TARGET].values)
    con.close()


init_db()


def confusion(db):
    r = db.execute(
        "SELECT COALESCE(SUM(actual=1 AND label='FRAUD'),0) AS tp, "
        "COALESCE(SUM(actual=0 AND label='FRAUD'),0) AS fp, "
        "COALESCE(SUM(actual=1 AND label='Legit'),0) AS fn, "
        "COALESCE(SUM(actual=0 AND label='Legit'),0) AS tn "
        "FROM predictions WHERE actual IS NOT NULL").fetchone()
    tp, fp, fn, tn = r["tp"], r["fp"], r["fn"], r["tn"]
    if tp + fp + fn + tn == 0:
        return None
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "recall": tp / (tp + fn) if tp + fn else None,
            "precision": tp / (tp + fp) if tp + fp else None}


# ---------- pages ----------
@app.route("/")
def index():
    db = get_db()
    s = db.execute("SELECT COUNT(*) AS total, SUM(label='FRAUD') AS fraud, "
                   "AVG(probability) AS avg_prob FROM predictions").fetchone()
    total, fraud = s["total"] or 0, s["fraud"] or 0
    recent = db.execute("SELECT id, probability, label FROM predictions "
                        "ORDER BY id DESC LIMIT 30").fetchall()[::-1]
    return render_template(
        "index.html", total=total, fraud=fraud, legit=total - fraud,
        rate=(fraud / total * 100) if total else 0, perf=confusion(db),
        chart_labels=[f"#{r['id']}" for r in recent],
        chart_values=[round(r["probability"], 4) for r in recent],
        chart_colors=["#dc3545" if r["label"] == "FRAUD" else "#198754" for r in recent],
        threshold=THRESHOLD)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    result, values, actual_raw = None, {}, ""
    if request.method == "POST":
        try:
            values = {c: request.form.get(c, "").strip() for c in COLUMNS}
            row = {c: float(v) for c, v in values.items()}  # raises if empty/invalid
            df = pd.DataFrame([row])
            probs, labels = score(df)
            actual_raw = request.form.get("actual", "")
            actual = int(actual_raw) if actual_raw in ("0", "1") else None
            save_rows(get_db(), df, probs, labels, "form",
                      None if actual is None else [actual])
            label = str(labels[0])
            outcome = None
            if actual is not None:
                outcome = {(1, "FRAUD"): ("Correct: fraud caught", "success"),
                           (0, "Legit"): ("Correct: legit transaction", "success"),
                           (0, "FRAUD"): ("False alarm: this was legit", "warning"),
                           (1, "Legit"): ("Missed fraud", "danger")}[(actual, label)]
            result = {"probability": float(probs[0]), "label": label,
                      "actual": actual, "outcome": outcome}
        except ValueError:
            flash("Every field must contain a valid number.", "danger")
    return render_template("predict.html", columns=COLUMNS, result=result,
                           values=values, actual_raw=actual_raw, threshold=THRESHOLD)


@app.route("/scan", methods=["GET", "POST"])
def scan():
    summary = None
    if request.method == "POST":
        n = request.form.get("n", 100, type=int)
        df = sample_demo(min(max(n, 10), 1000), request.form.get("fraud_rich") == "on")
        actual = df[TARGET].astype(int).values
        probs, labels = score(df)
        save_rows(get_db(), df, probs, labels, "scan", actual)
        flagged = labels == "FRAUD"
        tp = int(((actual == 1) & flagged).sum())
        fp = int(((actual == 0) & flagged).sum())
        fn = int(((actual == 1) & ~flagged).sum())
        summary = {"rows": len(df), "flagged": int(flagged.sum()),
                   "actual_fraud": int(actual.sum()), "tp": tp, "fp": fp, "fn": fn,
                   "tn": len(df) - tp - fp - fn}
        flash(f"Scanned {len(df)} real transactions from the dataset and saved the results.",
              "success")
    return render_template("scan.html", summary=summary, n_demo=len(DEMO),
                           n_fraud_demo=len(DEMO_FRAUD))


@app.route("/history")
def history():
    label = request.args.get("label", "all")
    page = max(request.args.get("page", 1, type=int), 1)
    where, args = "", []
    if label in ("FRAUD", "Legit"):
        where, args = "WHERE label = ?", [label]
    db = get_db()
    total = db.execute(f"SELECT COUNT(*) FROM predictions {where}", args).fetchone()[0]
    rows = db.execute(
        f"SELECT id, created_at, source, probability, label, actual FROM predictions {where} "
        "ORDER BY id DESC LIMIT ? OFFSET ?", args + [PER_PAGE, (page - 1) * PER_PAGE]
    ).fetchall()
    pages = max((total + PER_PAGE - 1) // PER_PAGE, 1)
    return render_template("history.html", rows=rows, label=label, page=page,
                           pages=pages, total=total)


@app.route("/transaction/<int:tid>")
def transaction(tid):
    row = get_db().execute("SELECT * FROM predictions WHERE id = ?", (tid,)).fetchone()
    if row is None:
        flash("Transaction not found.", "warning")
        return redirect(url_for("history"))
    return render_template("transaction.html", row=row,
                           features=json.loads(row["features"]), threshold=THRESHOLD)


@app.post("/delete/<int:tid>")
def delete(tid):
    db = get_db()
    db.execute("DELETE FROM predictions WHERE id = ?", (tid,))
    db.commit()
    flash(f"Deleted record #{tid}.", "info")
    return redirect(request.referrer or url_for("history"))


@app.post("/clear")
def clear():
    db = get_db()
    db.execute("DELETE FROM predictions")
    db.commit()
    flash("All records deleted.", "info")
    return redirect(url_for("history"))


@app.route("/export")
def export():
    rows = get_db().execute("SELECT * FROM predictions ORDER BY id").fetchall()
    out = pd.DataFrame([{
        "id": r["id"], "created_at": r["created_at"], "source": r["source"],
        "probability": r["probability"], "label": r["label"], "actual": r["actual"],
        **json.loads(r["features"])} for r in rows])
    return Response(out.to_csv(index=False), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=predictions.csv"})


@app.route("/model")
def model_info():
    return render_template("model.html", m=METRICS, n_features=len(COLUMNS),
                           threshold=THRESHOLD)


# ---------- REST API ----------
@app.get("/api/sample")
def api_sample():
    """A random real transaction from the built-in dataset (used by the form buttons)."""
    pool = {"fraud": DEMO_FRAUD, "legit": DEMO_LEGIT}.get(request.args.get("kind"), DEMO)
    row = pool.sample(1).iloc[0]
    return jsonify(features={c: float(row[c]) for c in COLUMNS},
                   actual=int(row[TARGET]), tx_id=int(row["tx_id"]))


@app.post("/api/predict")
def api_predict():
    data = request.get_json(silent=True) or {}
    feats = data.get("features")
    if not isinstance(feats, dict):
        return jsonify(error="Send JSON like {\"features\": {\"V1\": 0.1, ...}}"), 400
    missing = [c for c in COLUMNS if c not in feats]
    if missing:
        return jsonify(error="Missing features", missing=missing), 400
    try:
        df = pd.DataFrame([{c: float(feats[c]) for c in COLUMNS}])
    except (TypeError, ValueError):
        return jsonify(error="All feature values must be numbers"), 400
    probs, labels = score(df)
    save_rows(get_db(), df, probs, labels, "api")
    return jsonify(fraud_probability=float(probs[0]), prediction=str(labels[0]),
                   threshold=THRESHOLD)


@app.route("/health")
def health():
    return jsonify(status="ok", features=len(COLUMNS), dataset_rows=len(DEMO))


if __name__ == "__main__":
    app.run(debug=True, port=int(os.environ.get("PORT", 5000)))
