import copy

from tests.synthetic import SAMPLE_CUSTOMER


def test_health_reports_model_loaded(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok", "model_loaded": True}


def test_predict_returns_valid_prediction(client):
    r = client.post("/predict", json=SAMPLE_CUSTOMER)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["prediction"] in {"Churn", "Stay"}
    assert body["risk_level"] in {"low", "medium", "high"}
    assert "X-Process-Time-Ms" in r.headers


def test_risky_profile_scores_higher_than_loyal_profile(client):
    risky = {**SAMPLE_CUSTOMER, "Contract": "Month-to-month", "tenure": 1, "MonthlyCharges": 110}
    loyal = {**SAMPLE_CUSTOMER, "Contract": "Two year", "tenure": 70, "MonthlyCharges": 30}
    p_risky = client.post("/predict", json=risky).json()["churn_probability"]
    p_loyal = client.post("/predict", json=loyal).json()["churn_probability"]
    assert p_risky > p_loyal


def test_rejects_unknown_category(client):
    bad = {**SAMPLE_CUSTOMER, "Contract": "Weekly"}
    assert client.post("/predict", json=bad).status_code == 422


def test_rejects_missing_field(client):
    bad = copy.deepcopy(SAMPLE_CUSTOMER)
    del bad["tenure"]
    assert client.post("/predict", json=bad).status_code == 422


def test_rejects_negative_tenure_and_extra_fields(client):
    assert client.post("/predict", json={**SAMPLE_CUSTOMER, "tenure": -3}).status_code == 422
    assert client.post("/predict", json={**SAMPLE_CUSTOMER, "hacker": 1}).status_code == 422


def test_batch_prediction(client):
    r = client.post("/predict/batch", json={"customers": [SAMPLE_CUSTOMER] * 3})
    assert r.status_code == 200 and len(r.json()["predictions"]) == 3


def test_batch_rejects_empty_list(client):
    assert client.post("/predict/batch", json={"customers": []}).status_code == 422


def test_model_info_exposes_data_source(client):
    info = client.get("/model-info").json()
    assert info["selected_model"] in {"LogisticRegression", "RandomForest"}
    assert "data_source" in info


def test_returns_503_when_model_missing(client):
    saved = client.app.state.model
    client.app.state.model = None
    try:
        assert client.post("/predict", json=SAMPLE_CUSTOMER).status_code == 503
        assert client.get("/health").json()["model_loaded"] is False
    finally:
        client.app.state.model = saved
