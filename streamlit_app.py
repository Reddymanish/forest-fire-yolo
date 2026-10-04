"""Local Raspberry Pi Streamlit application for forest-fire inference."""

from __future__ import annotations

import hashlib
import io
import os
import platform
import shutil
import subprocess
import time
from datetime import datetime
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
DEMO_IMAGES_DIR = ROOT / "benchmark_images"
REPORT_DIR = Path(os.environ.get("REPORT_DIR", ROOT / "reports"))
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

st.set_page_config(page_title="Forest Fire Edge Detection", page_icon="🔥", layout="wide")


@st.cache_resource(show_spinner="Loading model on the Raspberry Pi CPU…")
def load_model(model_path: str) -> YOLO:
    return YOLO(model_path)


def read_temperature_c() -> float | None:
    """Read temperature from Linux thermal sysfs or Raspberry Pi vcgencmd."""
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


def demo_images() -> list[Path]:
    """Return the sorted sample images, including their class subfolders."""
    if not DEMO_IMAGES_DIR.is_dir():
        return []
    return sorted(
        (
            path for path in DEMO_IMAGES_DIR.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda path: path.as_posix().casefold(),
    )


def display_group(folder_name: str) -> str:
    return "No fire" if folder_name.casefold().replace(" ", "") == "nonfire" else folder_name.title()


def image_as_jpeg(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=92)
    return buffer.getvalue()


def send_to_workstation(report_path: Path) -> tuple[bool, str]:
    """Send a PDF from the Pi to the configured workstation over SSH/SCP."""
    target = os.environ.get("WORKSTATION_SCP_TARGET", "").strip()
    if not target:
        return False, "Set WORKSTATION_SCP_TARGET to user@workstation-host first."
    if not shutil.which("ssh") or not shutil.which("scp"):
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
        return False, "SSH/SCP timed out. Check the workstation connection."
    except subprocess.CalledProcessError as error:
        return False, (error.stderr or error.stdout or "SSH/SCP failed").strip()
    return True, f"Sent to {target}:{remote_dir}/"


def main() -> None:
    st.title("🔥 Forest Fire Edge Detection")
    st.caption("Images are analysed on this device's CPU. Download a PDF report or send it to your workstation.")

    with st.sidebar:
        st.header("Inference settings")
        model_path_text = st.text_input("Model path", value=os.environ.get("MODEL_PATH", str(DEFAULT_MODEL)))
        confidence = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
        image_size = st.select_slider("Image size", options=[320, 416, 512, 640], value=640)
        st.divider()
        temperature = read_temperature_c()
        st.metric("Device", "CPU only")
        st.metric("Pi temperature", f"{temperature:.1f} °C" if temperature is not None else "Unavailable")
        st.caption(platform.platform())

    model_path = Path(model_path_text).expanduser()
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    uploaded_image: Image.Image | None = None
    image_name = "image"
    source_id = ""

    upload_tab, demo_tab = st.tabs(["Upload image", "Demo benchmark images"])
    with upload_tab:
        uploaded = st.file_uploader(
            "Upload a scene image", type=["jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff"],
        )
        if uploaded is not None:
            upload_bytes = uploaded.getvalue()
            try:
                uploaded_image = Image.open(io.BytesIO(upload_bytes)).convert("RGB")
                image_name = Path(uploaded.name).name
                source_id = hashlib.sha256(upload_bytes).hexdigest()
            except Exception as error:
                st.error(f"Could not open the uploaded image: {error}")

    with demo_tab:
        examples = demo_images()
        if not examples:
            st.info("No demo images found. Add sample images beneath benchmark_images/<class-name>/.")
        else:
            st.caption("Choose an image from the Fire, Smoke, or No fire sample folders.")
            for start in range(0, len(examples), 3):
                columns = st.columns(3)
                for column, example_path in zip(columns, examples[start:start + 3]):
                    with column:
                        category = display_group(example_path.parent.name)
                        st.image(
                            str(example_path),
                            caption=f"{category} · {example_path.name}",
                            use_container_width=True,
                        )
                        if st.button("Use this image", key=f"demo_{example_path.as_posix()}", use_container_width=True):
                            st.session_state.selected_demo_image = str(example_path)

    selected_demo = st.session_state.get("selected_demo_image")
    if uploaded_image is None and selected_demo:
        demo_path = Path(selected_demo)
        if demo_path.is_file():
            try:
                with Image.open(demo_path) as demo_file:
                    uploaded_image = demo_file.convert("RGB")
                image_name = demo_path.name
                source_id = str(demo_path.resolve())
                st.info(f"Selected demo image: {display_group(demo_path.parent.name)} · {demo_path.name}")
            except Exception as error:
                st.error(f"Could not open the selected demo image: {error}")

    if not model_path.is_file():
        st.error(f"Model not found: {model_path}")
        st.info("Copy the trained checkpoint to models/best.pt or choose its path in the sidebar.")
        return

    if uploaded_image is None:
        st.info("Upload an image or choose one of the demo images to begin.")
        return

    run_key = f"{source_id}|{model_path.resolve()}|{confidence:.2f}|{image_size}"
    if st.session_state.get("last_run_key") != run_key:
        st.session_state.pop("last_run", None)

    left, right = st.columns(2)
    with left:
        st.image(uploaded_image, caption=f"Input image · {image_name}", use_container_width=True)
        run_clicked = st.button("Run inference", type="primary", use_container_width=True)

    if run_clicked:
        try:
            torch.set_num_threads(max(1, int(os.environ.get("TORCH_NUM_THREADS", "4"))))
            model = load_model(str(model_path))
            with st.spinner("Running inference on the CPU…"):
                started = time.perf_counter()
                result = model.predict(
                    uploaded_image,
                    conf=confidence,
                    imgsz=image_size,
                    device="cpu",
                    verbose=False,
                )[0]
                inference_ms = (time.perf_counter() - started) * 1000

            annotated = Image.fromarray(result.plot()[..., ::-1])
            detections = []
            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls.item())
                    names = result.names
                    label = names.get(class_id, str(class_id)) if isinstance(names, dict) else names[class_id]
                    detections.append({
                        "Class": label,
                        "Confidence": float(box.conf.item()),
                        "Bounding box (x1, y1, x2, y2)": [round(float(value), 1) for value in box.xyxy[0].tolist()],
                    })

            REPORT_DIR.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = REPORT_DIR / f"{Path(image_name).stem[:70]}_{timestamp}.pdf"
            create_report(uploaded_image, result, report_path, inference_ms, model_path)
            st.session_state.last_run_key = run_key
            st.session_state.last_run = {
                "annotated": image_as_jpeg(annotated),
                "detections": detections,
                "inference_ms": inference_ms,
                "report_path": str(report_path),
                "temperature_c": read_temperature_c(),
            }
        except Exception as error:
            st.error(f"Inference or report generation failed: {error}")

    run_data = st.session_state.get("last_run")
    if run_data and st.session_state.get("last_run_key") == run_key:
        with right:
            fps = 1000 / run_data["inference_ms"] if run_data["inference_ms"] > 0 else 0
            st.image(
                run_data["annotated"],
                caption=f"Inference: {run_data['inference_ms']:.1f} ms · {fps:.2f} FPS",
                use_container_width=True,
            )
            st.metric("Inference time", f"{run_data['inference_ms']:.1f} ms", f"{fps:.2f} FPS")
            if run_data["detections"]:
                st.dataframe(run_data["detections"], use_container_width=True, hide_index=True)
            else:
                st.info("No fire or smoke detections above the selected confidence threshold.")

        report_path = Path(run_data["report_path"])
        download_col, send_col = st.columns(2)
        with download_col:
            st.download_button(
                "Download PDF report",
                report_path.read_bytes(),
                file_name=report_path.name,
                mime="application/pdf",
                use_container_width=True,
            )
        with send_col:
            if st.button("Send PDF to workstation via SCP", use_container_width=True):
                success, message = send_to_workstation(report_path)
                (st.success if success else st.error)(message)
            if not os.environ.get("WORKSTATION_SCP_TARGET", "").strip():
                st.caption("Configure WORKSTATION_SCP_TARGET to enable SCP transfer.")
    else:
        with right:
            st.info("Run inference to view the annotated detections and report options here.")


if __name__ == "__main__":
    main()
