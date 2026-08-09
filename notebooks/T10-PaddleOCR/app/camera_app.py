#!/usr/bin/env python3
"""OpenCV camera preview for the PP-OCRv6 tiny DET-to-REC pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ocr_engine import DetectionConfig, OcrResult, PaddleOcrV6, RecognitionConfig


APP_DIR = Path(__file__).resolve().parent
TUTORIAL_DIR = APP_DIR.parent
DEFAULT_MODEL_DIR = TUTORIAL_DIR / "outputs" / "paddleocr_v6"
DEFAULT_DICTIONARY = APP_DIR / "assets" / "ppocrv6_tiny_dict.txt"
DEFAULT_FONT = APP_DIR / "assets" / "simfang.ttf"
CJK_FONT_CANDIDATES = (
    DEFAULT_FONT,
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJKjp-Regular.otf"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf"),
    Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"),
)
WINDOW_NAME = "PP-OCRv6 Tiny - DET + REC"


def parse_camera(value: str) -> int | str:
    return int(value) if value.isdecimal() else value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PP-OCRv6 tiny DET and REC models on a 640x480 camera stream."
    )
    parser.add_argument("--camera", "-c", default="/dev/video0", help="Camera index or device path")
    parser.add_argument("--width", type=int, default=640, help="Capture width (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="Capture height (default: 480)")
    parser.add_argument("--fps", type=float, default=15.0, help="Requested camera FPS (default: 15)")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--dictionary", type=Path, default=DEFAULT_DICTIONARY)
    parser.add_argument(
        "--font",
        type=Path,
        default=DEFAULT_FONT,
        help=f"CJK-capable .ttf/.ttc/.otf file (default: {DEFAULT_FONT.name})",
    )
    parser.add_argument("--font-size", type=int, default=18)
    parser.add_argument("--det-threshold", type=float, default=0.7)
    parser.add_argument("--box-threshold", type=float, default=0.6)
    parser.add_argument("--rec-threshold", type=float, default=0.5)
    parser.add_argument("--max-boxes", type=int, default=50)
    parser.add_argument("--mirror", action="store_true", help="Mirror the camera preview")
    parser.add_argument("--no-ort", action="store_true", help="Disable ORT for CPU subgraphs")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run one inference on every DET/REC model without opening a camera",
    )
    return parser.parse_args()


def open_camera(args: argparse.Namespace) -> cv2.VideoCapture:
    source = parse_camera(args.camera)
    backend = cv2.CAP_V4L2 if sys.platform.startswith("linux") else cv2.CAP_ANY
    capture = cv2.VideoCapture(source, backend)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera: {args.camera}")

    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    capture.set(cv2.CAP_PROP_FPS, args.fps)

    actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = capture.get(cv2.CAP_PROP_FPS)
    print(
        f"Camera: {args.camera} | requested={args.width}x{args.height}@{args.fps:g} "
        f"| actual={actual_width}x{actual_height}@{actual_fps:g}"
    )
    return capture


def resolve_cjk_font(requested: Path | None) -> Path:
    if requested is not None:
        path = requested.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Requested font was not found: {path}")
        return path

    for candidate in CJK_FONT_CANDIDATES:
        if candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(
        "No CJK-capable font was found. Install one with "
        "'sudo apt install fonts-noto-cjk', or pass "
        "'--font /path/to/a/CJK-font.ttf'."
    )


def load_font(
    requested: Path | None,
    size: int,
) -> tuple[ImageFont.FreeTypeFont, Path]:
    if size <= 0:
        raise ValueError("--font-size must be positive")
    path = resolve_cjk_font(requested)
    try:
        font = ImageFont.truetype(str(path), size=size)
    except OSError as error:
        raise RuntimeError(f"Could not load font: {path}") from error
    return font, path


def draw_results(
    frame_bgr: np.ndarray,
    result: OcrResult,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    display_fps: float,
) -> np.ndarray:
    canvas = frame_bgr.copy()
    recognized = {item.box_index: item for item in result.texts}

    for index, box in enumerate(result.boxes):
        color = (30, 220, 30) if index in recognized else (0, 190, 255)
        cv2.polylines(canvas, [np.rint(box).astype(np.int32)], True, color, 2, cv2.LINE_AA)

    image = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    for index, item in recognized.items():
        anchor_x = max(0, int(np.min(result.boxes[index][:, 0])))
        anchor_y = max(0, int(np.min(result.boxes[index][:, 1])))
        confidence = f"{item.score:.2f}"
        text_bounds = draw.textbbox((0, 0), item.text, font=font)
        confidence_bounds = draw.textbbox((0, 0), confidence, font=font)
        text_width = text_bounds[2] - text_bounds[0]
        label_height = max(
            text_bounds[3] - text_bounds[1],
            confidence_bounds[3] - confidence_bounds[1],
        )
        top = max(0, anchor_y - label_height - 4)
        draw.text((anchor_x, top), item.text, font=font, fill=(80, 255, 80))
        draw.text(
            (anchor_x + text_width + 8, top),
            confidence,
            font=font,
            fill=(255, 210, 60),
        )

    canvas = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    status = (
        f"FPS {display_fps:4.1f} | DET {result.detection_ms:5.1f} ms | "
        f"REC {result.recognition_ms:5.1f} ms | boxes {len(result.boxes)} | text {len(result.texts)}"
    )
    status_overlay = canvas[:28].copy()
    status_overlay[:] = (20, 20, 20)
    canvas[:28] = cv2.addWeighted(status_overlay, 0.45, canvas[:28], 0.55, 0.0)
    cv2.putText(canvas, status, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (235, 235, 235), 1, cv2.LINE_AA)
    return canvas


def main() -> int:
    args = parse_args()
    if args.width <= 0 or args.height <= 0 or args.fps <= 0:
        raise ValueError("Camera width, height, and FPS must be positive")
    if not 0.0 <= args.det_threshold <= 1.0:
        raise ValueError("--det-threshold must be between 0 and 1")
    if not 0.0 <= args.box_threshold <= 1.0:
        raise ValueError("--box-threshold must be between 0 and 1")
    if not 0.0 <= args.rec_threshold <= 1.0:
        raise ValueError("--rec-threshold must be between 0 and 1")

    engine = PaddleOcrV6(
        model_dir=args.model_dir,
        dictionary_path=args.dictionary,
        detection=DetectionConfig(
            pixel_threshold=args.det_threshold,
            box_threshold=args.box_threshold,
            max_boxes=args.max_boxes,
        ),
        recognition=RecognitionConfig(confidence_threshold=args.rec_threshold),
        use_ort=not args.no_ort,
    )

    if args.check:
        try:
            print(f"Model directory: {args.model_dir.expanduser().resolve()}")
            print(f"Dictionary     : {args.dictionary.expanduser().resolve()}")
            for message in engine.validate_models():
                print(message)
            print("PASS: PP-OCRv6 tiny DET-to-REC application contracts are valid.")
        finally:
            engine.close()
        return 0

    font, font_path = load_font(args.font, args.font_size)
    print(f"Text font: {font_path}")
    capture = open_camera(args)
    print("Press q or Esc in the preview window to stop.")

    smoothed_fps = 0.0
    try:
        while True:
            loop_started = perf_counter()
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError("Camera opened, but a frame could not be read")
            if frame.shape[1] != args.width or frame.shape[0] != args.height:
                frame = cv2.resize(frame, (args.width, args.height), interpolation=cv2.INTER_LINEAR)
            if args.mirror:
                frame = cv2.flip(frame, 1)

            result = engine.run(frame)
            elapsed = perf_counter() - loop_started
            instant_fps = 1.0 / elapsed if elapsed > 0 else 0.0
            smoothed_fps = instant_fps if smoothed_fps == 0 else smoothed_fps * 0.9 + instant_fps * 0.1
            preview = draw_results(frame, result, font, smoothed_fps)
            cv2.imshow(WINDOW_NAME, preview)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()
        engine.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
