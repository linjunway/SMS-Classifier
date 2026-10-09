def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True


def test_predict_schema(client):
    r = client.post("/predict", json={"text": "URGENT! Verify your account at http://bit.ly/x1"})
    assert r.status_code == 200
    body = r.json()
    assert body["label_name"] in {"Normal", "Spam", "Smishing"}
    assert set(body["probabilities"]) == {"Normal", "Spam", "Smishing"}
    assert body["signals"]["urls"] == 1


def test_predict_rejects_empty(client):
    assert client.post("/predict", json={"text": ""}).status_code == 422


def test_predict_rejects_too_long(client):
    assert client.post("/predict", json={"text": "a" * 1601}).status_code == 422


def test_batch(client):
    r = client.post("/predict/batch", json={"texts": ["hi there", "WIN £500 http://a.co"]})
    assert r.status_code == 200 and r.json()["count"] == 2


def test_batch_rejects_blank_and_oversize(client):
    assert client.post("/predict/batch", json={"texts": ["ok", "   "]}).status_code == 422
    assert client.post("/predict/batch", json={"texts": ["x"] * 101}).status_code == 422
