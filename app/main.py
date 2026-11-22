"""ShelfScan API (FastAPI).

Models load at startup in a background thread: /health answers at once, /ready stays
503 until both models are in memory. Uploads are decoded in memory and not stored; only
the annotated image is written (to SHELFSCAN_OUTPUT_DIR) so it can be fetched.

Run: uv run python run.py
"""

from __future__ import annotations

import logging
import re
import threading
import time
import uuid
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import Settings, get_settings
from app.engine import Busy, Engine, build_engine
from schemas.api import ErrorResponse, InferResponse
from services.images import ImageRejected, decode_upload
from services.visualization.visualizer import annotate

log = logging.getLogger("shelfscan")
OUTPUT_NAME = re.compile(r"^[0-9a-f]{32}\.jpg$")


class ApiError(Exception):
    def __init__(self, status: int, code: str, detail: str, headers: dict | None = None) -> None:
        self.status, self.code, self.detail, self.headers = status, code, detail, headers


def _error(request: Request, status: int, code: str, detail: str, headers=None) -> JSONResponse:
    rid = getattr(request.state, "request_id", "-")
    body = ErrorResponse(error=code, detail=detail, request_id=rid).model_dump()
    return JSONResponse(body, status_code=status, headers=headers)


def create_app(settings: Settings | None = None, engine: Engine | None = None) -> FastAPI:
    settings = settings or get_settings()
    state = {"engine": engine, "loading": "ready" if engine else "starting", "failed": False}

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if state["engine"] is None:

            def load() -> None:
                state["loading"] = "loading the detector and the embedder"
                t0 = time.perf_counter()
                try:
                    state["engine"] = build_engine(settings)
                    state["loading"] = "ready"
                    log.info("models loaded in %.1f s", time.perf_counter() - t0)
                except Exception:
                    state["failed"] = True
                    state["loading"] = "model loading failed; see the server log"
                    log.exception("model loading failed")

            threading.Thread(target=load, name="model-loader", daemon=True).start()
        yield

    app = FastAPI(
        title="ShelfScan",
        version="0.2.0",
        summary="Detects products on a shelf photo and groups the ones that look alike.",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def request_id(request: Request, call_next):
        request.state.request_id = uuid.uuid4().hex[:16]
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError):
        return _error(request, exc.status, exc.code, exc.detail, exc.headers)

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError):
        first = exc.errors()[0] if exc.errors() else {}
        where = ".".join(str(p) for p in first.get("loc", []) if p not in ("body",))
        msg = first.get("msg", "invalid request")
        return _error(request, 422, "invalid_request", f"{where}: {msg}" if where else msg)

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException):
        code = {404: "not_found", 405: "method_not_allowed"}.get(exc.status_code, "http_error")
        return _error(request, exc.status_code, code, str(exc.detail))

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, exc: Exception):
        log.exception("unhandled error")
        return _error(request, 500, "internal_error", "Something went wrong; it was logged.")

    def ready_engine() -> Engine:
        if state["engine"] is None:
            raise ApiError(503, "not_ready", f"The service is starting ({state['loading']}).")
        return state["engine"]

    @app.get("/health", tags=["ops"])
    def health():
        """Liveness: the process is up."""
        return {"status": "ok"}

    @app.get("/ready", tags=["ops"])
    def ready(request: Request):
        """Readiness: both models are loaded."""
        engine = state["engine"]
        if engine is None:
            code = "startup_failed" if state["failed"] else "not_ready"
            return _error(request, 503, code, f"The service is starting ({state['loading']}).")
        return {"status": "ready", "models": engine.models, "torch_threads": settings.torch_threads}

    def infer(request: Request, file: UploadFile) -> InferResponse:
        t0 = time.perf_counter()
        engine = ready_engine()
        limit = int(settings.max_upload_mb * 1024 * 1024)
        data = file.file.read(limit + 1)
        if len(data) > limit:
            raise ApiError(
                413, "too_large", f"Uploads are limited to {settings.max_upload_mb:g} MB."
            )
        try:
            decoded = decode_upload(data, settings.max_pixels, settings.min_side_px)
        except ImageRejected as exc:
            raise ApiError(exc.status, exc.code, exc.detail) from None
        t_decoded = time.perf_counter()
        try:
            detections, timings = engine.run(decoded.image)
        except Busy:
            raise ApiError(
                503,
                "busy",
                "Another image is being processed; try again shortly.",
                {"Retry-After": "30"},
            ) from None
        t_annotate = time.perf_counter()
        name = f"{uuid.uuid4().hex}.jpg"
        out_dir = Path(settings.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        annotate(decoded.image, detections).save(out_dir / name, "JPEG", quality=90)
        t_end = time.perf_counter()

        counts = Counter(d["group_id"] for d in detections)
        return InferResponse(
            request_id=request.state.request_id,
            image={
                "format": decoded.format,
                "width": decoded.width,
                "height": decoded.height,
                "exif_orientation": decoded.exif_orientation,
            },
            detections=detections,
            products=len(detections),
            groups=[{"group_id": g, "count": c} for g, c in counts.most_common()],
            annotated_image_url=f"/v1/outputs/{name}",
            timings={
                "decode_ms": round((t_decoded - t0) * 1000, 1),
                "detect_ms": round(timings["detect_ms"], 1),
                "group_ms": round(timings["group_ms"], 1),
                "annotate_ms": round((t_end - t_annotate) * 1000, 1),
                "total_ms": round((t_end - t0) * 1000, 1),
            },
            models=engine.models,
            settings={
                "confidence_threshold": settings.confidence_threshold,
                "nms_iou": settings.nms_iou,
                "similarity_threshold": settings.similarity_threshold,
                "min_crop_px": settings.min_crop_px,
                "torch_threads": settings.torch_threads,
            },
        )

    responses = {code: {"model": ErrorResponse} for code in (413, 415, 422, 503)}

    @app.post("/v1/infer", response_model=InferResponse, responses=responses, tags=["inference"])
    def infer_v1(request: Request, file: UploadFile = File(..., description="Shelf photo")):  # noqa: B008
        """Detect products, group the ones that look alike, return JSON and an annotated image."""
        return infer(request, file)

    @app.post("/infer", response_model=InferResponse, deprecated=True, tags=["inference"])
    def infer_legacy(request: Request, file: UploadFile = File(...)):  # noqa: B008
        """The January endpoint's path; same as /v1/infer."""
        return infer(request, file)

    @app.get("/v1/outputs/{name}", tags=["inference"], response_class=FileResponse)
    def output(name: str):
        """An annotated image from a previous /v1/infer call."""
        path = Path(settings.output_dir) / name
        if not OUTPUT_NAME.match(name) or not path.is_file():
            raise ApiError(404, "not_found", "No such annotated image.")
        return FileResponse(path, media_type="image/jpeg")

    return app
