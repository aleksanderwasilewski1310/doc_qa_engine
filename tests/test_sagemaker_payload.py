"""Tests for the SageMaker vision endpoint request and response handling."""

import base64
import json
from io import BytesIO
from types import SimpleNamespace

from app.main import invoke_sagemaker_vision


def test_invoke_sagemaker_vision_uses_raw_image_bytes(monkeypatch):
    """Send the image bytes as base64 and return the extracted text."""
    captured = {}

    def fake_invoke_endpoint(**kwargs):
        captured.update(kwargs)
        return {"Body": BytesIO(b'{"extracted_text": "hello"}')}

    monkeypatch.setattr(
        "app.main.SAGEMAKER_RUNTIME",
        SimpleNamespace(invoke_endpoint=fake_invoke_endpoint),
    )
    monkeypatch.setattr("app.main.SAGEMAKER_VISION_ENDPOINT", "test-endpoint")

    result = invoke_sagemaker_vision(b"fake-image")

    assert result == "hello"
    assert captured["EndpointName"] == "test-endpoint"
    assert captured["ContentType"] == "application/json"
    payload = json.loads(captured["Body"])
    assert payload["inputs"] == base64.b64encode(b"fake-image").decode("utf-8")
