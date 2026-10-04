"""Create a compact PDF report for one Ultralytics detection result."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from fpdf import FPDF
from PIL import Image


def create_report(
    source_image: Image.Image,
    result: Any,
    output_path: str | Path,
    inference_ms: float,
    model_path: str | Path,
) -> Path:
    """Save a PDF containing the annotated image, detections and timing."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotated = Image.fromarray(result.plot()[..., ::-1])
    image_path = output_path.with_suffix(".annotated.jpg")
    annotated.save(image_path, quality=92)

    names = result.names
    boxes = result.boxes
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=14)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 11, "Forest Fire Detection Report", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=10)
    pdf.cell(0, 7, f"Generated: {datetime.now().astimezone().isoformat(timespec='seconds')}", new_x="LMARGIN", new_y="NEXT")
    pdf.multi_cell(
        pdf.epw,
        6,
        f"Image: {source_image.width} x {source_image.height} pixels\n"
        f"Model: {Path(model_path).name}\n"
        f"Inference: {inference_ms:.2f} ms",
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.ln(2)
    if boxes is None or len(boxes) == 0:
        pdf.cell(0, 7, "No detections above the selected confidence threshold.", new_x="LMARGIN", new_y="NEXT")
    else:
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, f"Detections ({len(boxes)}):", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", size=10)
        for index, box in enumerate(boxes, start=1):
            class_id = int(box.cls.item())
            label = names.get(class_id, str(class_id)) if isinstance(names, dict) else names[class_id]
            confidence = float(box.conf.item())
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            pdf.multi_cell(
                pdf.epw,
                6,
                f"{index}. {label}  |  confidence {confidence:.1%}  |  "
                f"box ({x1:.0f}, {y1:.0f}) to ({x2:.0f}, {y2:.0f})",
                new_x="LMARGIN",
                new_y="NEXT",
            )
    pdf.ln(3)
    with Image.open(image_path) as image:
        width, height = image.size
    available_w = pdf.w - pdf.l_margin - pdf.r_margin
    available_h = pdf.h - pdf.get_y() - pdf.b_margin
    scale = min(available_w / width, available_h / height)
    if scale > 0:
        pdf.image(str(image_path), w=width * scale, h=height * scale)
    pdf.output(str(output_path))
    image_path.unlink(missing_ok=True)
    return output_path
