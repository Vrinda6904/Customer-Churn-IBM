import os

import pytest
from fastapi.testclient import TestClient

from tests.synthetic import make_dataset


@pytest.fixture(scope="session")
def client(tmp_path_factory):
    from train import train

    model_dir = tmp_path_factory.mktemp("model")
    train(make_dataset(1500), str(model_dir), data_source="synthetic")
    os.environ["MODEL_PATH"] = str(model_dir / "churn_model.joblib")

    from app.main import app
    import app.main as main_module
    main_module.MODEL_PATH = model_dir / "churn_model.joblib"
    with TestClient(app) as c:  # runs lifespan -> loads model
        yield c
