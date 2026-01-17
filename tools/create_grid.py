from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

def create_before_after_grid(
    upload_image_path: str,
    annotated_image_path: str,
    output_path: str,
    label_height: int = 50,
    bg_color=(255, 255, 255)
):
    """
    Create a side-by-side grid image showing input and output images
    with labels above them.

    Args:
        upload_image_path (str): Path to input image.
        annotated_image_path (str): Path to annotated output image.
        output_path (str): Path to save the combined grid image.
        label_height (int): Height reserved for labels.
        bg_color (tuple): Background color.
    """

    # Load images
    input_img = Image.open(upload_image_path).convert("RGB")
    output_img = Image.open(annotated_image_path).convert("RGB")

    # Resize output image to match input (safety)
    output_img = output_img.resize(input_img.size)

    width, height = input_img.size

    # Create canvas
    grid_width = width * 2
    grid_height = height + label_height

    canvas = Image.new("RGB", (grid_width, grid_height), bg_color)
    draw = ImageDraw.Draw(canvas)

    # Paste images
    canvas.paste(input_img, (0, label_height))
    canvas.paste(output_img, (width, label_height))

    # Load default font
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 24)
    except:
        font = ImageFont.load_default()

    # Draw labels
    draw.text((width // 2 - 30, 10), "INPUT", fill=(0, 0, 0), font=font)
    draw.text((width + width // 2 - 40, 10), "OUTPUT", fill=(0, 0, 0), font=font)

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)

    return str(output_path)

if __name__ == "__main__":
    upload_img = (
        "data/uploads/"
        "57c175cacce04cd89b305aa64c06f25b_Good_Quality-1993_8.jpg"
    )

    annotated_img = (
        "data/outputs/"
        "annotated_57c175cacce04cd89b305aa64c06f25b_Good_Quality-1993_8.jpg"
    )

    grid_output = (
        "data/outputs/"
        "before_after_grid_57c175ca.jpg"
    )

    create_before_after_grid(
        upload_image_path=upload_img,
        annotated_image_path=annotated_img,
        output_path=grid_output
    )

    print("Grid image created successfully!")
