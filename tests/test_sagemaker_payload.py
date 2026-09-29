import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import invoke_sagemaker_vision


def test_invoke_sagemaker_vision_uses_raw_image_bytes(monkeypatch):
    captured = {}

    class FakeRuntime:
        def invoke_endpoint(self, **kwargs):
            captured.update(kwargs)
            return {"Body": lambda: None}

    monkeypatch.setattr("app.main.sagemaker_runtime", FakeRuntime())
    monkeypatch.setattr("app.main.SAGEMAKER_VISION_ENDPOINT", "test-endpoint")

    class FakeBody:
        def read(self):
            return b'{"extracted_text": "hello"}'

    class FakeResponse:
        Body = FakeBody()

    monkeypatch.setattr("app.main.sagemaker_runtime", FakeRuntime())

    def fake_invoke_endpoint(**kwargs):
        captured.update(kwargs)
        return {"Body": FakeBody()}

    monkeypatch.setattr("app.main.sagemaker_runtime", type("Runtime", (), {"invoke_endpoint": fake_invoke_endpoint})())

    result = invoke_sagemaker_vision(b"fake-image")

    assert result == "hello"
    assert captured["EndpointName"] == "test-endpoint"
    assert captured["ContentType"] == "application/json"
    assert b'"name": "image"' in captured["Body"]
