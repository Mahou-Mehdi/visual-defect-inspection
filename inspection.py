"""A transparent patch-statistics baseline for image anomaly inspection."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw

SIZE = 128
PATCH = 8
EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def image_files(folder: Path) -> list[Path]:
    files = sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if not files:
        raise ValueError(f"No supported images in {folder}")
    return files


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        if image.width > 8192 or image.height > 8192:
            raise ValueError("Images larger than 8192 pixels per side are unsupported.")
        return image.convert("RGB").resize((SIZE, SIZE), Image.Resampling.BILINEAR)


def features(image: Image.Image) -> np.ndarray:
    gray = np.asarray(image.convert("L").resize((SIZE, SIZE)), dtype=np.float64) / 255.0
    gy, gx = np.gradient(gray)
    channels = [gray, np.abs(gx), np.abs(gy)]
    result = []
    for channel in channels:
        cells = channel.reshape(SIZE // PATCH, PATCH, SIZE // PATCH, PATCH).transpose(0, 2, 1, 3)
        result.extend([cells.mean(axis=(2, 3)), cells.std(axis=(2, 3))])
    return np.stack(result, axis=-1)


def anomaly_map(descriptors: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(((descriptors - mean) / scale) ** 2, axis=-1))


def image_score(heat: np.ndarray) -> float:
    # Averaging the two most unusual patches is less sensitive to one isolated noisy cell.
    return float(np.partition(heat.ravel(), -2)[-2:].mean())


def fit(train: list[Path], calibration: list[Path], output: Path, percentile: float = 95) -> dict:
    if output.suffix.lower() != ".npz":
        raise ValueError("Model output must use the .npz extension.")
    if len(train) < 2 or len(calibration) < 5:
        raise ValueError("Use at least 2 training images and 5 separate calibration images.")
    if not 50 <= percentile < 100:
        raise ValueError("Calibration percentile must be in [50, 100).")
    train_hashes, val_hashes = [digest(p) for p in train], [digest(p) for p in calibration]
    if set(train_hashes) & set(val_hashes):
        raise ValueError("Identical images appear in training and calibration splits.")
    if len(set(train_hashes)) != len(train_hashes) or len(set(val_hashes)) != len(val_hashes):
        raise ValueError("Repeated image content within training/calibration splits.")
    descriptors = np.stack([features(read_image(path)) for path in train])
    mean = descriptors.mean(axis=0)
    scale = np.maximum(descriptors.std(axis=0, ddof=1), 0.005)
    scores = np.array([image_score(anomaly_map(features(read_image(p)), mean, scale)) for p in calibration])
    threshold = float(np.percentile(scores, percentile, method="higher"))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, mean=mean, scale=scale, threshold=threshold,
                        training_hashes=np.array(train_hashes), calibration_hashes=np.array(val_hashes),
                        calibration_percentile=percentile)
    return {"train_normal_count": len(train), "calibration_normal_count": len(calibration),
            "calibration_percentile": percentile, "threshold": threshold}


def load_model(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as source:
        model = {key: source[key].copy() for key in source.files}
    if model["mean"].shape != (16, 16, 6) or model["scale"].shape != (16, 16, 6):
        raise ValueError("Model shape is incompatible with this version.")
    if not np.isfinite(model["mean"]).all() or not np.isfinite(model["scale"]).all() or np.any(model["scale"] <= 0):
        raise ValueError("Invalid model values.")
    if not np.isfinite(model["threshold"]).all():
        raise ValueError("Invalid model threshold.")
    return model


def infer(path: Path, model: dict) -> tuple[float, np.ndarray, Image.Image]:
    image = read_image(path)
    heat = anomaly_map(features(image), model["mean"], model["scale"])
    return image_score(heat), heat, image


def overlay(image: Image.Image, heat: np.ndarray, threshold: float) -> Image.Image:
    # This is an anomaly-strength visualization, not a pixel probability or segmentation mask.
    strength = np.clip(heat / max(2 * threshold, 1e-8), 0, 1)
    expanded = Image.fromarray((strength * 255).astype(np.uint8)).resize(image.size, Image.Resampling.BILINEAR)
    alpha = np.asarray(expanded, dtype=np.float64)[..., None] / 255 * 0.75
    rgb = np.asarray(image, dtype=np.float64)
    tint = np.full_like(rgb, (255, 35, 40))
    return Image.fromarray(np.uint8(np.clip(rgb * (1 - alpha) + tint * alpha, 0, 255)))


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positives, negatives = scores[labels == 1], scores[labels == 0]
    if len(positives) == 0 or len(negatives) == 0:
        raise ValueError("AUROC requires both normal and defective samples.")
    # Rank statistic with tie handling; bounded memory for larger user datasets.
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=float)
    sorted_scores = scores[order]
    first = 0
    while first < len(scores):
        last = first + 1
        while last < len(scores) and sorted_scores[last] == sorted_scores[first]:
            last += 1
        ranks[order[first:last]] = (first + 1 + last) / 2
        first = last
    return float((ranks[labels == 1].sum() - len(positives) * (len(positives) + 1) / 2) / (len(positives) * len(negatives)))


def evaluate(data: Path, model_path: Path, output: Path) -> dict:
    model = load_model(model_path)
    normal, defective = image_files(data / "test/good"), image_files(data / "test/defect")
    files = normal + defective
    used_hashes = set(model["training_hashes"]) | set(model["calibration_hashes"])
    test_hashes = [digest(path) for path in files]
    if used_hashes & set(test_hashes):
        raise ValueError("Test images overlap the training/calibration data by file hash.")
    if len(set(test_hashes)) != len(test_hashes):
        raise ValueError("Repeated image content in the test split would distort evaluation.")
    labels, scores, timings = np.array([0] * len(normal) + [1] * len(defective)), [], []
    examples = []
    for index, path in enumerate(files):
        started = time.perf_counter()
        score, heat, image = infer(path, model)
        timings.append((time.perf_counter() - started) * 1000)
        scores.append(score)
        if index in (0, len(normal)):
            examples.append((path, score, image, overlay(image, heat, float(model["threshold"]))))
    scores = np.array(scores)
    predicted = scores > model["threshold"]
    tp, fp = int(np.sum(predicted & (labels == 1))), int(np.sum(predicted & (labels == 0)))
    threshold = float(model["threshold"])
    summary = {
        "method": "per-location diagonal Gaussian patch statistics", "image_size": [SIZE, SIZE], "patch_size": PATCH,
        "training_normal_images": len(model["training_hashes"]), "calibration_normal_images": len(model["calibration_hashes"]),
        "test_normal_images": len(normal), "test_defective_images": len(defective),
        "threshold": threshold, "calibration_percentile": float(model["calibration_percentile"]),
        "image_auroc": auc(labels, scores), "true_positives": tp, "false_positives": fp,
        "true_negatives": len(normal) - fp, "false_negatives": len(defective) - tp,
        "defect_recall": tp / len(defective), "normal_false_alarm_rate": fp / len(normal),
        "mean_inference_ms": float(np.mean(timings)), "p95_inference_ms": float(np.percentile(timings, 95)),
        "timing_scope": "CPU image read, resize, feature extraction and scoring; excludes model load and overlays",
        "predictions": [{"image": p.relative_to(data).as_posix(), "defective": bool(label),
                         "score": float(score), "predicted_defective": bool(pred)}
                        for p, label, score, pred in zip(files, labels, scores, predicted)],
    }
    provenance = data / "dataset.json"
    if provenance.exists():
        summary["dataset"] = json.loads(provenance.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    canvas = Image.new("RGB", (590, 390), "#101827")
    draw = ImageDraw.Draw(canvas)
    draw.text((18, 12), "Patch-statistics inspection | original + anomaly overlay", fill="white")
    draw.text((18, 30), "Red = unusual local appearance; score above threshold triggers alert", fill="#a8b9cf")
    for row, (path, score, original, overlaid) in enumerate(examples):
        top = 65 + 155 * row
        canvas.paste(original, (18, top))
        canvas.paste(overlaid, (160, top))
        draw.text((310, top + 20), path.parent.name.upper(), fill="white")
        draw.text((310, top + 40), f"Score: {score:.2f}", fill="white")
        draw.text((310, top + 60), f"Threshold: {threshold:.2f}", fill="white")
        draw.text((310, top + 80), "ALERT" if score > threshold else "NORMAL", fill="#fb923c" if score > threshold else "#34d399")
    canvas.save(output / "inspection-demo.png")
    print(json.dumps({k: v for k, v in summary.items() if k != "predictions"}, indent=2))
    return summary


def synthetic_image(rng: np.random.Generator, defect: bool) -> Image.Image:
    y, x = np.mgrid[:SIZE, :SIZE]
    texture = (125 + rng.normal(0, 2) + 9 * np.sin(x * 0.75 + rng.uniform(-0.6, 0.6))
               + 6 * np.sin(y * 0.9 + rng.uniform(-0.6, 0.6)) + rng.normal(0, 3, (SIZE, SIZE)))
    image = Image.fromarray(np.uint8(np.clip(texture, 0, 255))).convert("RGB")
    if defect:
        draw = ImageDraw.Draw(image)
        left, top = [int(v) for v in rng.integers(15, 90, size=2)]
        color = int(rng.choice([65, 195]))
        if rng.random() < 0.5:
            draw.line((left, top, left + int(rng.integers(12, 30)), top + int(rng.integers(15, 30))), fill=(color,) * 3, width=int(rng.integers(2, 5)))
        else:
            radius = int(rng.integers(5, 11))
            draw.ellipse((left, top, left + 2 * radius, top + 2 * radius), fill=(color,) * 3)
    return image


def generate(destination: Path, seed: int = 42):
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Dataset destination must be empty; use a fresh directory.")
    rng = np.random.default_rng(seed)
    for split, count, defect in (("train/good", 64, False), ("val/good", 32, False),
                                 ("test/good", 64, False), ("test/defect", 64, True)):
        folder = destination / split
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            synthetic_image(rng, defect).save(folder / f"{i:03d}.png")
    (destination / "dataset.json").write_text(json.dumps({"kind": "original synthetic textures", "seed": seed,
        "license": "MIT", "description": "Procedural woven grayscale textures; bright/dark scratches and spots. No real industrial images."}, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    generator = sub.add_parser("generate", help="Create original, explicitly synthetic image data")
    generator.add_argument("--data", type=Path, default=Path("data/synthetic"))
    generator.add_argument("--seed", type=int, default=42)
    train = sub.add_parser("train", help="Fit normal-only statistics and calibrate on separate normals")
    train.add_argument("--data", type=Path, required=True)
    train.add_argument("--model", type=Path, default=Path("output/model.npz"))
    train.add_argument("--percentile", type=float, default=95)
    test = sub.add_parser("evaluate", help="Evaluate untouched good and defective test images")
    test.add_argument("--data", type=Path, required=True)
    test.add_argument("--model", type=Path, default=Path("output/model.npz"))
    test.add_argument("--output", type=Path, default=Path("output/evaluation"))
    predict = sub.add_parser("predict", help="Score one image and export an anomaly overlay")
    predict.add_argument("image", type=Path)
    predict.add_argument("--model", type=Path, default=Path("output/model.npz"))
    predict.add_argument("--output", type=Path, default=Path("output/overlay.png"))
    args = parser.parse_args()
    try:
        if args.command == "generate":
            generate(args.data, args.seed)
            print(f"Created synthetic dataset at {args.data}")
        elif args.command == "train":
            report = fit(image_files(args.data / "train/good"), image_files(args.data / "val/good"), args.model, args.percentile)
            print(json.dumps(report, indent=2))
        elif args.command == "evaluate":
            evaluate(args.data, args.model, args.output)
        else:
            model = load_model(args.model)
            score, heat, image = infer(args.image, model)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            overlay(image, heat, float(model["threshold"])).save(args.output)
            print(json.dumps({"score": score, "threshold": float(model["threshold"]), "defective": bool(score > model["threshold"])}, indent=2))
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
