import uuid
from pathlib import Path

from flask import Blueprint, jsonify, request
from werkzeug.utils import secure_filename

from services.detection.detector import detect_products
from services.grouping.grouper import group_products
from services.visualization.visualizer import visualize

api = Blueprint("api", __name__)


@api.route("/infer", methods=["POST"])
def infer():
    if "file" not in request.files:
        return jsonify({"status": "error", "message": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"status": "error", "message": "No selected file"}), 400

    filename = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{filename}"

    project_root = Path(__file__).parent.parent
    upload_dir = project_root / "data" / "uploads"
    output_dir = project_root / "data" / "outputs"

    upload_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_path = upload_dir / unique_name
    file.save(str(save_path))

    # -------- PIPELINE STARTS HERE --------

    detections = detect_products(str(save_path))
    grouped_detections = group_products(detections, str(save_path))
    output_image_path = visualize(
        image_path=str(save_path), detections=grouped_detections, output_dir=str(output_dir)
    )

    # -------- PIPELINE ENDS HERE --------

    response = {
        "status": "success",
        "input_image": str(save_path.relative_to(project_root)),
        "output_image": str(Path(output_image_path).relative_to(project_root)),
        "detections": grouped_detections,
    }

    return jsonify(response)


def register_routes(app):
    app.register_blueprint(api)
