"""Simple local Streamlit UI for Raspberry Pi forest-fire inference."""

from __future__ import annotations

import hashlib
import io
import os
import shutil
import subprocess
import time
from pathlib import Path

import streamlit as st
import torch
from PIL import Image
from ultralytics import YOLO

from pdf_report import create_report


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "models" / "best.pt"
if not DEFAULT_MODEL.exists():
    DEFAULT_MODEL = ROOT / "final_outputs" / "detect" / "train" / "weights" / "best.pt"
MODEL_PATH = Path(os.environ.get("MODEL_PATH", DEFAULT_MODEL))
REPORT_DIR = Path(os.environ.get("REPORT_DIR", ROOT / "reports"))
DEMO_DIR = ROOT / "benchmark_images"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

st.set_page_config(page_title="Forest Fire Detector", page_icon="🔥", layout="wide")


@st.cache_resource(show_spinner="Loading the model on the Pi CPU…")
def load_model(model_path: str) -> YOLO:
    return YOLO(model_path)


def read_temperature_c() -> float | None:
    """Read the Pi temperature from sysfs, falling back to vcgencmd."""
    for sensor in Path("/sys/class/thermal").glob("thermal_zone*/temp"):
        try:
            value = float(sensor.read_text().strip())
            return value / 1000 if value > 1000 else value
        except (OSError, ValueError):
            continue
    try:
        output = subprocess.check_output(["vcgencmd", "measure_temp"], text=True, timeout=2)
        return float(output.split("=", 1)[1].split("'", 1)[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def demo_groups() -> dict[str, list[Path]]:
    """Find the sample images supplied under benchmark_images/<class>/."""
    if not DEMO_DIR.is_dir():
        return {}
    groups = {}
    for folder in sorted(path for path in DEMO_DIR.iterdir() if path.is_dir()):
        images = sorted(
            path for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS
        )
        if images:
            display_name = "No fire" if folder.name.lower() == "non fire" else folder.name.title()
            groups[display_name] = images
    return groups


def image_to_jpeg(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=92)
    return buffer.getvalue()


def send_report(report_path: Path) -> tuple[bool, str]:
    """Send a report from the Pi to the workstation with SSH and SCP."""
    target = os.environ.get("WORKSTATION_SCP_TARGET", "").strip()
    if not target:
        return False, "Set WORKSTATION_SCP_TARGET to user@workstation-host first."
    if not shutil.which("scp") or not shutil.which("ssh"):
        return False, "OpenSSH is missing. Install it on the Pi with: sudo apt install openssh-client"

    remote_dir = os.environ.get("WORKSTATION_REPORT_DIR", "~/forest-fire-reports")
    try:
        subprocess.run(
            ["ssh", "-o", "BatchMode=yes", target, "mkdir", "-p", remote_dir],
            check=True, capture_output=True, text=True, timeout=30,
        )
        subprocess.run(
            ["scp", "-o", "BatchMode=yes", str(report_path), f"{target}:{remote_dir}/"],
            check=True, capture_output=True, text=True, timeout=120,
        )
    except subprocess.TimeoutExpired:
        return False, "SSH/SCP timed out. Check that the workstation is reachable."
    except subprocess.CalledProcessError as error:
        return False, (error.stderr or error.stdout or "SSH/SCP failed").strip()
    return True, f"Report sent to {target}:{remote_dir}/"


def main() -> None:
    st.title("Forest Fire Detector")
    st.caption("Choose an image. The Raspberry Pi runs the model locally and creates a PDF report.")

    if not MODEL_PATH.is_file():
        st.error(f"Model file not found: {MODEL_PATH}")
        st.info("Copy the trained checkpoint to models/best.pt, or set the MODEL_PATH environment variable.")
        st.stop()

    temp_c = read_temperature_c()
    status = st.columns(3)
    status[0].metric("Inference device", "CPU only")
    status[1].metric("Model", MODEL_PATH.name)
    status[2].metric("Temperature", f"{temp_c:.1f} °C" if temp_c is not None else "Unavailable")

    source = st.radio("Image source", ["Upload an image", "Try a demo image"], horizontal=True)
    image: Image.Image | None = None
    image_name = "image"
    source_id = ""

    if source == "Upload an image":
        uploaded = st.file_uploader("Choose a photo", type=["jpg", "jpeg", "png", "bmp", "webp"])
        if uploaded is not None:
            image_bytes = uploaded.getvalue()
            try:
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                image_name = Path(uploaded.name).name
                source_id = hashlib.sha256(image_bytes).hexdigest()
            except Exception as error:
                st.error(f"Could not open that image: {error}")
    else:
        groups = demo_groups()
        if not groups:
            st.info("No demo images found. Add JPG or PNG images under benchmark_images/<class-name>/.")
        else:
            group_name = st.selectbox("Demo group", list(groups))
            paths = groups[group_name]
            demo_path = st.selectbox("Choose a sample", paths, format_func=lambda path: path.name)
            with Image.open(demo_path) as sample:
                image = sample.convert("RGB")
            image_name = demo_path.name
            source_id = str(demo_path.resolve())
            st.image(image, caption=f"Demo image · {group_name}", use_container_width=True)

    with st.expander("Detection settings", expanded=False):
        confidence = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
        image_size = st.select_slider("Inference image size", options=[320, 416, 512, 640], value=640)
        st.caption("Larger images can improve small-object detection, but take longer on the Pi.")

    run_key = f"{source_id}|{confidence:.2f}|{image_size}"
    if st.session_state.get("last_run_key") != run_key:
        st.session_state.pop("last_run", None)

    if image is not None:
        if source == "Upload an image":
            st.image(image, caption="Selected image", use_container_width=True)
        if st.button("Run detection", type="primary", use_container_width=True):
            try:
                torch.set_num_threads(max(1, int(os.environ.get("TORCH_NUM_THREADS", "4"))))
                model = load_model(str(MODEL_PATH))
                start = time.perf_counter()
                result = model.predict(
                    source=image, device="cpu", conf=confidence, imgsz=image_size, verbose=False,
                )[0]
                inference_ms = (time.perf_counter() - start) * 1000
                annotated = Image.fromarray(result.plot()[..., ::-1])

                detections = []
                if result.boxes is not None:
                    for box in result.boxes:
                        class_id = int(box.cls.item())
                        names = result.names
                        label = names.get(class_id, str(class_id)) if isinstance(names, dict) else names[class_id]
                        detections.append({
                            "Class": label,
                            "Confidence": f"{float(box.conf.item()):.1%}",
                        })

                REPORT_DIR.mkdir(parents=True, exist_ok=True)
                report_path = REPORT_DIR / f"{Path(image_name).stem[:70]}_{time.strftime('%Y%m%d_%H%M%S')}.pdf"
                create_report(image, result, report_path, inference_ms, MODEL_PATH)
                st.session_state.last_run_key = run_key
                st.session_state.last_run = {
                    "image_name": image_name,
                    "original": image_to_jpeg(image),
                    "annotated": image_to_jpeg(annotated),
                    "inference_ms": inference_ms,
                    "detections": detections,
                    "report_path": str(report_path),
                    "temperature_c": read_temperature_c(),
                }
            except Exception as error:
                st.error(f"Inference or report generation failed: {error}")

    run_data = st.session_state.get("last_run")
    if run_data and st.session_state.get("last_run_key") == run_key:
        st.divider()
        st.subheader("Result")
        original_col, result_col = st.columns(2)
        with original_col:
            st.image(run_data["original"], caption="Original", use_container_width=True)
        with result_col:
            st.image(run_data["annotated"], caption="Detections", use_container_width=True)

        metrics = st.columns(3)
        metrics[0].metric("Inference time", f"{run_data['inference_ms']:.1f} ms")
        metrics[1].metric("Objects found", str(len(run_data["detections"])))
        current_temp = run_data["temperature_c"]
        metrics[2].metric("Pi temperature", f"{current_temp:.1f} °C" if current_temp is not None else "Unavailable")

        if run_data["detections"]:
            st.dataframe(run_data["detections"], hide_index=True, use_container_width=True)
        else:
            st.info("No fire or smoke detections above this confidence threshold.")

        report_path = Path(run_data["report_path"])
        action_col, send_col = st.columns(2)
        with action_col:
            st.download_button(
                "Download PDF report", report_path.read_bytes(), file_name=report_path.name,
                mime="application/pdf", use_container_width=True,
            )
        with send_col:
            if st.button("Send PDF to workstation", use_container_width=True):
                success, message = send_report(report_path)
                (st.success if success else st.error)(message)
            if not os.environ.get("WORKSTATION_SCP_TARGET", "").strip():
                st.caption("Set WORKSTATION_SCP_TARGET to enable SCP transfer.")


if __name__ == "__main__":
    main()
