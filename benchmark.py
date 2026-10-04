"""Benchmark CPU inference on the 30 curated images in benchmark_images/."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
from ultralytics import YOLO


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "models" / "best.pt"
if not DEFAULT_MODEL.exists():
    DEFAULT_MODEL = ROOT / "final_outputs" / "detect" / "train" / "weights" / "best.pt"
CLASS_FOLDERS = {
    "Fire": {"fire"},
    "Smoke": {"smoke"},
    "No fire": {"nofire", "nonfire"},
}
IMAGES_PER_CLASS = 10


def read_temperature_c() -> float | None:
    """Read a CPU temperature from Linux thermal sysfs or vcgencmd."""
    for thermal_file in Path("/sys/class/thermal").glob("thermal_zone*/temp"):
        try:
            value = float(thermal_file.read_text().strip())
            return value / 1000.0 if value > 1000 else value
        except (OSError, ValueError):
            continue
    try:
        output = subprocess.check_output(["vcgencmd", "measure_temp"], text=True, timeout=2)
        return float(output.split("=", 1)[1].split("'", 1)[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def normalized_folder_name(name: str) -> str:
    return "".join(character for character in name.lower() if character.isalnum())


def select_benchmark_images(images_dir: Path) -> list[tuple[str, Path]]:
    """Select exactly 10 sorted images from each of the three demo groups."""
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Benchmark image folder not found: {images_dir}")

    selected: list[tuple[str, Path]] = []
    found_groups: set[str] = set()
    for folder in sorted(path for path in images_dir.iterdir() if path.is_dir()):
        normalized = normalized_folder_name(folder.name)
        class_name = next(
            (label for label, aliases in CLASS_FOLDERS.items() if normalized in aliases),
            None,
        )
        if class_name is None:
            continue
        if class_name in found_groups:
            raise ValueError(f"More than one folder maps to the '{class_name}' group.")

        images = sorted(
            (path for path in folder.iterdir()
             if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
            key=lambda path: path.name.casefold(),
        )
        if len(images) < IMAGES_PER_CLASS:
            raise ValueError(
                f"Expected at least {IMAGES_PER_CLASS} images in {folder}; found {len(images)}."
            )
        # If the directory is also used for the UI gallery, keep the benchmark
        # stable: only its first ten alphabetically sorted samples are measured.
        selected.extend((class_name, path) for path in images[:IMAGES_PER_CLASS])
        found_groups.add(class_name)

    missing = set(CLASS_FOLDERS) - found_groups
    if missing:
        raise ValueError(f"Missing image folder(s) for: {', '.join(sorted(missing))}.")
    if len(selected) != 30:
        raise ValueError(f"Expected exactly 30 selected images; selected {len(selected)}.")
    return selected


def percentile(values: list[float], percentage: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int((percentage / 100) * len(ordered) + 0.999999) - 1))
    return ordered[index]


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p95": percentile(values, 95),
        "min": min(values),
        "max": max(values),
    }


def detected_class_counts(result: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    if result.boxes is None:
        return counts
    names = result.names
    for box in result.boxes:
        class_id = int(box.cls.item())
        name = names.get(class_id, str(class_id)) if isinstance(names, dict) else names[class_id]
        counts[name] = counts.get(name, 0) + 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Ultralytics model checkpoint")
    parser.add_argument("--images-dir", default=str(ROOT / "benchmark_images"), help="Folder containing Fire, Smoke and No fire subfolders")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size (default: 640)")
    parser.add_argument("--threads", type=int, default=4, help="PyTorch CPU thread count (default: 4)")
    parser.add_argument("--warmup", type=int, default=3, help="Unmeasured warm-up inferences (default: 3)")
    parser.add_argument("--output", default="benchmark_results.json", help="JSON results file")
    parser.add_argument("--csv-output", default="benchmark_results.csv", help="Per-image CSV results file")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.is_file():
        parser.error(f"Model checkpoint not found: {model_path}")
    if args.imgsz < 1 or args.threads < 1 or args.warmup < 0:
        parser.error("--imgsz and --threads must be positive; --warmup cannot be negative")
    try:
        selected_images = select_benchmark_images(Path(args.images_dir))
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    print(f"Selected {len(selected_images)} images (10 each: Fire, Smoke, No fire).")
    print(f"Loading model: {model_path}")
    torch.set_num_threads(args.threads)
    model = YOLO(str(model_path))
    first_image = selected_images[0][1]
    for warmup_run in range(args.warmup):
        model.predict(str(first_image), device="cpu", imgsz=args.imgsz, verbose=False)
        print(f"Warm-up {warmup_run + 1}/{args.warmup}")

    temperature_start = read_temperature_c()
    records: list[dict[str, Any]] = []
    latencies_ms: list[float] = []
    sampled_temperatures: list[float] = []

    print("Running one measured inference per selected image…")
    for index, (expected_group, image_path) in enumerate(selected_images, start=1):
        start = time.perf_counter()
        result = model.predict(str(image_path), device="cpu", imgsz=args.imgsz, verbose=False)[0]
        elapsed_ms = (time.perf_counter() - start) * 1000
        temperature_c = read_temperature_c()
        if temperature_c is not None:
            sampled_temperatures.append(temperature_c)

        prediction_counts = detected_class_counts(result)
        detection_total = sum(prediction_counts.values())
        record = {
            "image": str(image_path),
            "group": expected_group,
            "inference_ms": round(elapsed_ms, 3),
            "detections": detection_total,
            "detected_classes": prediction_counts,
            "temperature_c": round(temperature_c, 2) if temperature_c is not None else None,
        }
        records.append(record)
        latencies_ms.append(elapsed_ms)
        temperature_text = f"{temperature_c:.1f} °C" if temperature_c is not None else "n/a"
        print(
            f"{index:02}/30  {expected_group:7}  {image_path.name}  "
            f"{elapsed_ms:.1f} ms  {detection_total} detection(s)  {temperature_text}"
        )

    temperature_end = read_temperature_c()
    report = {
        "device": platform.platform(),
        "processor": platform.processor(),
        "device_used": "cpu",
        "model": str(model_path),
        "images_dir": str(Path(args.images_dir)),
        "selected_image_count": len(records),
        "images_per_group": IMAGES_PER_CLASS,
        "warmup_runs": args.warmup,
        "imgsz": args.imgsz,
        "torch_threads": args.threads,
        "latency_ms": summarize(latencies_ms),
        "latency_ms_by_group": {
            group: summarize([row["inference_ms"] for row in records if row["group"] == group])
            for group in CLASS_FOLDERS
        },
        "temperature_c": {
            "start": temperature_start,
            "mean_during_run": statistics.mean(sampled_temperatures) if sampled_temperatures else None,
            "max_during_run": max(sampled_temperatures) if sampled_temperatures else None,
            "end": temperature_end,
        },
        "per_image": records,
    }

    json_path = Path(args.output)
    csv_path = Path(args.csv_output)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["image", "group", "inference_ms", "detections", "detected_classes", "temperature_c"],
        )
        writer.writeheader()
        for record in records:
            csv_record = record.copy()
            csv_record["detected_classes"] = json.dumps(csv_record["detected_classes"], ensure_ascii=False)
            writer.writerow(csv_record)

    print("\nOverall CPU inference latency (milliseconds):")
    for name, value in report["latency_ms"].items():
        print(f"  {name:>6}: {value:.2f}")
    if sampled_temperatures:
        print(
            f"Temperature: start={temperature_start if temperature_start is not None else 'n/a'} °C, "
            f"mean={statistics.mean(sampled_temperatures):.1f} °C, "
            f"max={max(sampled_temperatures):.1f} °C, "
            f"end={temperature_end if temperature_end is not None else 'n/a'} °C"
        )
    else:
        print("Temperature sensor unavailable on this system.")
    print(f"Saved per-image and summary JSON: {json_path}")
    print(f"Saved per-image CSV: {csv_path}")


if __name__ == "__main__":
    main()
