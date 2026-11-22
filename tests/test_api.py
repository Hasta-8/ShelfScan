"""The API through FastAPI's TestClient, with fake models."""

import hashlib
import threading
import time

import pytest
from fastapi.testclient import TestClient
from helpers import FakeDetector, FakeGrouper, encode, shelf_image
from PIL import Image

import app.main as main
from app.config import Settings
from app.engine import Engine


@pytest.fixture
def settings(tmp_path):
    return Settings(
        _env_file=None, output_dir=tmp_path / "out", max_upload_mb=1, busy_timeout_s=0.2
    )


@pytest.fixture
def make_client(settings):
    def make(detector=None, grouper=None, **overrides):
        s = settings.model_copy(update=overrides)
        engine = Engine(
            detector or FakeDetector(), grouper or FakeGrouper(), s, {"detector": "fake"}
        )
        # Return 500 responses instead of re-raising, as a real client would see them.
        return TestClient(main.create_app(s, engine), raise_server_exceptions=False)

    return make


def infer(client, data, name="shelf.jpg", path="/v1/infer"):
    return client.post(path, files={"file": (name, data, "image/jpeg")})


def test_infer_returns_the_contract(make_client):
    r = infer(make_client(), encode(shelf_image()))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["products"] == 3 and body["image"] == {
        "format": "JPEG",
        "width": 640,
        "height": 480,
        "exif_orientation": 1,
    }
    assert body["detections"][0] == {
        "bbox": [40, 100, 140, 300],
        "confidence": 0.9,
        "label": "product",
        "group_id": 1,
    }
    assert {g["group_id"] for g in body["groups"]} == {0, 1, 2}
    assert set(body["timings"]) == {"decode_ms", "detect_ms", "group_ms", "annotate_ms", "total_ms"}
    assert body["request_id"] == r.headers["x-request-id"]
    assert body["settings"]["torch_threads"] == 4
    assert isinstance(body["settings"]["torch_threads"], int)


def test_annotated_image_is_served_and_deterministic(make_client):
    client = make_client()
    urls = [infer(client, encode(shelf_image())).json()["annotated_image_url"] for _ in range(2)]
    assert urls[0] != urls[1]  # a new file per request
    images = [client.get(u) for u in urls]
    assert all(r.status_code == 200 and r.headers["content-type"] == "image/jpeg" for r in images)
    digests = {hashlib.sha256(r.content).hexdigest() for r in images}
    assert len(digests) == 1  # same input -> byte-identical output


def test_output_route_refuses_anything_but_its_own_names(make_client):
    client = make_client()
    for name in ["../app/config.py", "x.jpg", "a" * 32 + ".png"]:
        assert client.get(f"/v1/outputs/{name}").status_code == 404


def test_exif_rotated_upload_is_processed_upright(make_client):
    detector = FakeDetector()
    client = make_client(detector=detector)
    stored = shelf_image().transpose(Image.Transpose.ROTATE_90)
    body = infer(client, encode(stored, exif_orientation=6)).json()
    assert detector.seen == [(640, 480)]  # the model saw the upright image
    assert body["image"]["exif_orientation"] == 6 and body["image"]["width"] == 640


@pytest.mark.parametrize(
    ("data", "name", "status", "code"),
    [
        (b"hello", "notes.txt", 415, "unsupported_type"),
        (b"\x89PNG\r\n\x1a\n" + b"0" * 100, "fake.png", 422, "corrupt_image"),
        (encode(shelf_image(60, 60)), "tiny.jpg", 422, "too_small"),
        (b"", "empty.jpg", 422, "empty_file"),
    ],
)
def test_bad_uploads_get_clear_4xx_errors(make_client, data, name, status, code):
    r = infer(make_client(), data, name=name)
    assert r.status_code == status and r.json()["error"] == code
    assert set(r.json()) == {"error", "detail", "request_id"}


def test_oversized_upload_is_413(make_client):
    r = infer(make_client(max_upload_mb=0.01), encode(shelf_image(2000, 1500)))
    assert r.status_code == 413 and r.json()["error"] == "too_large"


def test_missing_file_field_is_422(make_client):
    r = make_client().post("/v1/infer", data={"other": "x"})
    assert r.status_code == 422 and r.json()["error"] == "invalid_request"


def test_legacy_path_still_works(make_client):
    assert infer(make_client(), encode(shelf_image()), path="/infer").status_code == 200


def test_ready_is_503_until_models_load(settings, monkeypatch):
    release = threading.Event()
    fake = Engine(FakeDetector(), FakeGrouper(), settings, {"detector": "fake"})

    def slow_build(_settings):
        release.wait(5)
        return fake

    monkeypatch.setattr(main, "build_engine", slow_build)
    with TestClient(main.create_app(settings)) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/ready").status_code == 503
        r = infer(client, encode(shelf_image()))
        assert r.status_code == 503 and r.json()["error"] == "not_ready"
        release.set()
        for _ in range(100):
            if client.get("/ready").status_code == 200:
                break
            time.sleep(0.02)
        assert client.get("/ready").json()["status"] == "ready"
        assert infer(client, encode(shelf_image())).status_code == 200


def test_failed_model_loading_is_reported_without_details(settings, monkeypatch):
    def broken(_settings):
        raise RuntimeError("/home/secret/models missing")

    monkeypatch.setattr(main, "build_engine", broken)
    with TestClient(main.create_app(settings)) as client:
        for _ in range(100):
            r = client.get("/ready")
            if r.json()["error"] == "startup_failed":
                break
            time.sleep(0.02)
        assert r.status_code == 503 and "secret" not in r.text


def test_a_second_request_waits_then_gets_busy(make_client):
    gate = threading.Event()

    class SlowDetector(FakeDetector):
        def detect(self, image, *a, **k):
            gate.wait(5)
            return super().detect(image, *a, **k)

    client = make_client(detector=SlowDetector())
    first = {}
    t = threading.Thread(target=lambda: first.update(r=infer(client, encode(shelf_image()))))
    t.start()
    time.sleep(0.1)
    second = infer(client, encode(shelf_image()))  # busy_timeout_s = 0.2
    assert second.status_code == 503 and second.json()["error"] == "busy"
    assert second.headers["retry-after"] == "30"
    gate.set()
    t.join(5)
    assert first["r"].status_code == 200


def test_unexpected_errors_hide_details(make_client):
    class Broken(FakeDetector):
        def detect(self, *a, **k):
            raise ValueError("/home/secret/path exploded")

    r = infer(make_client(detector=Broken()), encode(shelf_image()))
    assert r.status_code == 500 and r.json()["error"] == "internal_error"
    assert "secret" not in r.text
