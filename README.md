# Titanic Survival Predictor

A small end-to-end machine learning practice project:

1. **`training/`** — trains a model that predicts whether a Titanic
   passenger would have survived, based on their details.
2. **`app/backend/`** — a FastAPI web service that loads the trained model
   and answers prediction requests.
3. **`app/frontend/`** — a simple web page with a form where you enter
   passenger details and see the prediction.

This is a learning project, kept intentionally small and simple.

## 1. Train the model

```bash
cd training
python -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
python train.py
```

This will:
- Download the classic Titanic dataset (via the `seaborn` library — no manual downloading needed)
- Clean it up (fill in missing ages/ports)
- Train a Random Forest classifier
- Print an accuracy score, confusion matrix, and classification report
- Save the trained model to `app/backend/model/titanic_model.joblib`

You should see output ending with something like:

```
Accuracy: 0.816
...
Saved trained model to .../app/backend/model/titanic_model.joblib
```

You need to run this step once before the backend will work, since the
backend loads that saved model file.

## 2. Start the backend API

```bash
cd app/backend
python -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

Leave this running in its own terminal. You can check it's alive by
opening http://localhost:8000/health in a browser — it should return
`{"status":"ok","model_loaded":true}`.

You can also try it directly with `curl`:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"pclass":3,"sex":"male","age":22,"sibsp":1,"parch":0,"fare":7.25,"embarked":"S"}'
```

## 3. Open the frontend

With the backend still running, just open `app/frontend/index.html` directly
in your web browser (double-click it, or drag it into a browser window —
no server needed). Fill in the form and click **Predict**.

If you deploy the backend somewhere other than `localhost:8000`, update the
`API_URL` constant near the top of the `<script>` tag in `index.html`.

## Project structure

```
training/
  train.py            # loads data, trains + evaluates the model, saves it
  requirements.txt
app/
  backend/
    main.py            # FastAPI app: /health and /predict endpoints
    requirements.txt
    model/              # trained model gets saved here (git-ignored)
  frontend/
    index.html          # the form + JS that calls the backend
```

## Notes for learning

- The dataset comes from `seaborn.load_dataset("titanic")`, a built-in copy
  of the classic Kaggle Titanic dataset — no downloads or API keys needed.
- The trained model file (`*.joblib`) is git-ignored since it's a generated
  build artifact, not source code — anyone can regenerate it by running
  `train.py`.
- The backend and frontend are deliberately minimal (no database, no auth,
  no deployment config) since this is a practice project, not a production
  app.
