# FraudGuard – Credit Card Fraud Detection Web App

A full-stack machine learning web app that scores credit card transactions for fraud, stores every prediction in a database, and shows a live dashboard.

**Stack:** Python · scikit-learn (Logistic Regression) · Flask · SQLite · Bootstrap · Chart.js · Gunicorn

## Features
- **Dashboard** – totals, fraud rate, fraud vs legit chart, recent probabilities
- **Check Transaction** – load a real transaction from the built-in dataset (or type your own) and get a fraud probability, then see whether the model was right
- **Scan Dataset** – one click scores up to 1,000 real transactions and compares the results with the true labels (no file upload needed)
- **Database UI** – browse, filter (Fraud / Legit), paginate, view details, see actual vs predicted, delete, export to CSV
- **Model page** – precision, recall, F1, ROC-AUC, confusion matrix and curves
- **REST API** – `POST /api/predict`

## 1. Run locally

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements-train.txt

# The model in model/ is already trained on the 284,807-transaction dataset.
# To retrain (optional), put creditcard_fraud.csv in this folder and run:
python train.py --train creditcard_fraud.csv

python app.py                     # open http://127.0.0.1:5000
```

The dataset is already built in (`data/demo_transactions.csv.gz`), so the website never asks for an upload. On first launch the database is pre-filled with 300 scanned transactions so the dashboard isn't empty (set `SEED_DEMO=0` to turn that off).

Your CSV needs a `Class` column (0 = legit, 1 = fraud); every other column is used as a feature.

## 2. Deploy (free) on Render

1. Make sure `model/`, `data/` (the built-in dataset, <1 MB) and `static/*.png` are committed. **Do not commit the big CSV** (it's in `.gitignore`).
2. `requirements.txt` pins `scikit-learn==1.8.0`, the version the model was saved with. If you retrain locally with a different version, update that pin to match (`pip freeze | grep -i scikit-learn`), otherwise the saved model may not load on the server.
3. Push to a GitHub repo:
   ```bash
   git init && git add . && git commit -m "Fraud detection app"
   git branch -M main
   git remote add origin https://github.com/<you>/fraud-detection-app.git
   git push -u origin main
   ```
4. On [render.com](https://render.com): **New → Web Service** → connect the repo.
   Build command: `pip install -r requirements.txt` · Start command: `gunicorn app:app`
   (or use **New → Blueprint**, which reads `render.yaml`).
5. Open the URL Render gives you. Done.

> **Note:** Render's free tier has an ephemeral disk, so the SQLite database resets on each redeploy/restart. That's fine for a demo. For persistent data, add a Render disk and set the `DB_PATH` environment variable, or switch to PostgreSQL.
> Also, free instances sleep after inactivity, so the first load can take ~30–60 seconds. Mention this if you share the link.

Other hosts work too: Railway, Fly.io, PythonAnywhere (`Procfile` included).

## 3. API example

```bash
curl -X POST https://<your-app>/api/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"Time": 0, "V1": -1.35, ... , "Amount": 149.62}}'
```
Response:
```json
{"fraud_probability": 0.0003, "prediction": "Legit", "threshold": 0.9}
```

## Project structure
```
app.py               Flask app, routes, SQLite, REST API
train.py             trains the model and writes model/ + static/ files
templates/           HTML (Jinja2 + Bootstrap)
model/               saved model and metrics (created by train.py)
data/                built-in dataset: 5,000 real held-out transactions, 98 of them fraud (created by train.py)
requirements.txt     runtime dependencies
Procfile, render.yaml   deployment config
```

## Resume bullets (numbers from the held-out test set; re-check them on the Model page if you retrain)
- Built and deployed an end-to-end credit card fraud detection web app (Flask, scikit-learn, SQLite) that scores transactions in real time and stores results in a database with a dashboard, history browser and CSV export.
- Trained a class-weighted logistic regression on 284,807 transactions (0.17% fraud), tuning the decision threshold on a validation set to reach **85% recall** and **65% precision** (ROC-AUC 0.97, F1 0.73) on a held-out test set, versus roughly 20% precision at the default-style threshold.
- Exposed the model via a REST API and batch CSV scoring; deployed on Render with Gunicorn.

## Dataset
The full Kaggle file is ~100 MB, which is too big for GitHub, so the repo bundles a compressed sample of 5,000 transactions taken only from the held-out test split (all 98 test-set frauds plus 4,902 legit ones). That keeps the live accuracy shown in the app honest, because the model never trained on those rows.

Designed for the public Kaggle "Credit Card Fraud Detection" dataset (columns `Time`, `V1`–`V28`, `Amount`, `Class`), but works with any numeric CSV that has a `Class` column.
