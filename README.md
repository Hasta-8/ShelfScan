# ShelfScan

**Scalable Retail Shelf Product Detection and Grouping Pipeline**

## Overview

ShelfScan is a modular AI pipeline designed to analyze retail shelf images. Given a single shelf image, the system:

- Detects all visible products
- Groups visually similar products (brand-level approximation)
- Produces structured JSON output and an annotated image

The solution is built with scalability, modularity, and real-world constraints in mind, particularly CPU-only inference environments.

## Problem Context

Retail shelf images present several challenges:

- Dense product placement
- Overlapping and visually similar SKUs
- Multiple brands appearing within the same shelf row
- High variation in lighting, packaging, and viewpoints

Simple spatial heuristics (for example, grouping by row position) fail in realistic shelves. ShelfScan addresses this by combining robust object detection with visual embedding–based grouping.

## High-Level Architecture

The system follows a service-oriented pipeline:

```
Input Image
     |
     v
Detection Service (DETR)
     |
     v
Grouping Service (Visual Embeddings)
     |
     v
Visualization Service
     |
     v
JSON Response + Annotated Image
```

Each component is independent and can be replaced or scaled without affecting the rest of the system.

## Project Structure

```
ShelfScan/
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── main.py
│   └── routes.py
│
├── services/
│   ├── detection/
│   │   ├── __init__.py
│   │   └── detector.py
│   │
│   ├── grouping/
│   │   ├── __init__.py
│   │   └── grouper.py
│   │
│   └── visualization/
│       ├── __init__.py
│       └── visualizer.py
│
├── schemas/
│   ├── __init__.py
│   └── response_schema.py
│
├── data/
│   ├── uploads/
│   └── outputs/
│
├── run.py
├── requirements.txt
└── README.md
```

## Detection Module

### Model

**DETR (ResNet-50)** trained on **SKU-110K**

- Source: Hugging Face

### Rationale

- SKU-110K is specifically curated for dense retail shelf images
- Transformer-based detection handles cluttered scenes better than anchor-based detectors
- High recall is prioritized to avoid missing products

### Post-processing

- Confidence thresholding
- Lightweight Non-Maximum Suppression (NMS) to remove duplicate detections

## Grouping Module

### Motivation

Retail shelves often contain multiple brands in the same shelf row, making spatial grouping unreliable.

### Approach

1. Crop detected product bounding boxes
2. Extract visual embeddings using a self-supervised vision model
3. Cluster embeddings using cosine similarity

### Model

**DINOv2-small** (`facebook/dinov2-small`)

#### Why DINOv2?

- Strong visual representation learning
- Lightweight and CPU-friendly
- No authentication or gated access
- Well-suited for image-to-image similarity

### Outcome

- Products with similar packaging and branding are grouped together
- Mixed-brand shelf rows are handled correctly
- Grouping is robust to small visual variations

## Visualization Module

- Draws bounding boxes on the original image
- Assigns a unique color per group
- Saves annotated output image to `data/outputs/`

This enables easy qualitative inspection of both detection and grouping quality.

## API Interface

### Endpoint

**POST /infer**

### Request

- Multipart form-data
- Field: `file` (input shelf image)

### Response (JSON)

```json
{
  "status": "success",
  "input_image": "data/uploads/example.jpg",
  "output_image": "data/outputs/annotated_example.jpg",
  "detections": [
    {
      "bbox": [x1, y1, x2, y2],
      "confidence": 0.92,
      "group_id": 2,
      "label": "product"
    }
  ]
}
```

## How to Run

### 1. Create and Activate Virtual Environment

```bash
python -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the Server

```bash
python run.py
```

### 4. Run Inference

```bash
curl -X POST http://127.0.0.1:5000/infer \
  -F "file=@path/to/shelf_image.jpg"
```

## Scalability Considerations

- Stateless API design
- Detection, grouping, and visualization are decoupled
- Models are loaded once at startup
- Easy to:
  - Swap models
  - Split services
  - Add GPU acceleration
- Suitable for batch or microservice-based extensions

## Limitations

- Grouping is based on visual similarity, not SKU identification
- No text-based brand recognition or OCR
- CPU inference is slower than GPU-based systems
- Price tags and shelf labels are ignored

## Future Improvements

- SKU-level classification
- OCR-based brand verification
- GPU acceleration
- Batch inference support
- Integration with inventory or planogram systems