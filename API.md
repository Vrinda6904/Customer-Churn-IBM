# Churn Prediction API

A FastAPI service that serves the churn model from `CustomerChurnIBM.ipynb`, packaged with Docker
and tested in CI.

## Architecture

```
client --> FastAPI (app/main.py) --> validation (app/schemas.py, Pydantic)
                |                              |
                |                    422 on bad input
                v
        sklearn Pipeline (model/churn_model.joblib)
        [StandardScaler + OneHotEncoder] -> [LogisticRegression | RandomForest]
```

`train.py` fits both models from the notebook, picks the better one by ROC-AUC, and saves one
Pipeline, so the API accepts raw customer fields and applies the exact same preprocessing as training.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness + whether the model loaded |
| GET | `/model-info` | selected model, metrics, data source, threshold |
| POST | `/predict` | churn probability for one customer |
| POST | `/predict/batch` | up to 1000 customers |
| GET | `/docs` | interactive Swagger UI |

```bash
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' -d '{
  "gender":"Female","SeniorCitizen":0,"Partner":"Yes","Dependents":"No","tenure":5,
  "PhoneService":"Yes","MultipleLines":"No","InternetService":"Fiber optic",
  "OnlineSecurity":"No","OnlineBackup":"No","DeviceProtection":"No","TechSupport":"No",
  "StreamingTV":"Yes","StreamingMovies":"Yes","Contract":"Month-to-month",
  "PaperlessBilling":"Yes","PaymentMethod":"Electronic check","MonthlyCharges":89.5,
  "TotalCharges":447.5,"numAdminTickets":1,"numTechTickets":2}'
# {"churn_probability":0.81,"prediction":"Churn","risk_level":"high"}
```

## Run it

```bash
pip install -r requirements-dev.txt
python train.py --data Customer_Churn_Dataset.xlsx    # writes model/churn_model.joblib
python -m pytest -q
uvicorn app.main:app --reload                         # http://localhost:8000/docs

docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

Train with the same pinned `scikit-learn` version used in `requirements.txt`. A model pickled with
a different version can fail to load inside the container.

## Deploy free on Render

1. Commit `model/churn_model.joblib` and `model/metrics.json` (the dataset itself can stay out of git).
2. Push to GitHub, then on render.com choose New > Blueprint (reads `render.yaml`) or New > Web Service > Docker.
3. Pick the Free instance. The service sleeps after 15 idle minutes, so the first request can take about a minute.

## Results

Evaluated on a stratified 20% hold-out (1,409 of 7,043 customers; 26.5% churn rate):

| Model | Accuracy | ROC-AUC | Churn precision | Churn recall |
|---|---|---|---|---|
| Logistic Regression (selected) | 85.7% | 0.928 | 0.76 | 0.68 |
| Random Forest | 84.2% | 0.908 | 0.76 | 0.60 |

- Served predictions match direct pipeline output on 300 sampled real customers (0 mismatches).
- Median `/predict` latency: [X] ms (measure yours: run the container, then check the `X-Process-Time-Ms` header).
- 10 automated tests; CI builds the image and smoke-tests the running container on every push.
