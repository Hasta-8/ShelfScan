import random
from pathlib import Path

from PIL import Image, ImageDraw


def visualize(image_path: str, detections: list, output_dir: str) -> str:
    """
    Visualize detections on image, save annotated image.
    Args:
            image_path (str): Path to input image.
            detections (list): List of detection dicts (with group_id).
            output_dir (str): Directory to save annotated image.
    Returns:
            str: Path to saved annotated image.
    """
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Assign a color to each group_id
    group_ids = {det.get("group_id", 0) for det in detections}
    color_map = {gid: tuple(random.choices(range(50, 256), k=3)) for gid in group_ids}

    for det in detections:
        bbox = det["bbox"]
        group_id = det.get("group_id", 0)
        color = color_map.get(group_id, (255, 0, 0))
        draw.rectangle(bbox, outline=color, width=3)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / ("annotated_" + Path(image_path).name)
    img.save(str(out_path))
    return str(out_path)
