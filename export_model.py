"""
export_model.py — Export YOLOv8n to TFLite for Android ADAS App
================================================================
Run this script ONCE on your PC (NOT on the phone or Pi).

Requirements:
    pip install ultralytics

Output:
    yolov8n_float32.tflite  ← copy this to android_app/app/src/main/assets/yolov8n.tflite

Usage:
    python export_model.py
"""

import shutil
from pathlib import Path


def export_yolov8_tflite(
    model_name: str = "yolov8n.pt",
    imgsz: int = 320,
    use_int8: bool = False,
) -> Path:
    """
    Download and export YOLOv8n to TFLite format.

    Parameters
    ----------
    model_name : str
        YOLOv8 model variant. 'yolov8n.pt' is the smallest/fastest.
    imgsz : int
        Input resolution. Must match INPUT_SIZE in InferenceEngine.kt (320).
    use_int8 : bool
        If True, produce an INT8 quantized model (faster on Android, ~4× smaller).
        Requires a calibration dataset — set to False for a quick float32 export.

    Returns
    -------
    Path
        Path to the exported .tflite file.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit(
            "ultralytics not installed.\n"
            "Run: pip install ultralytics"
        )

    print(f"[1/3] Loading model: {model_name}  (downloads ~6 MB if not cached)")
    model = YOLO(model_name)

    print(f"[2/3] Exporting to TFLite  (imgsz={imgsz}, int8={use_int8}) ...")
    export_kwargs = dict(format="tflite", imgsz=imgsz)
    if use_int8:
        export_kwargs["int8"] = True

    model.export(**export_kwargs)

    # Ultralytics saves alongside the weights, e.g. yolov8n_saved_model/yolov8n_float32.tflite
    suffix = "int8" if use_int8 else "float32"
    tflite_path = Path(f"yolov8n_saved_model/yolov8n_{suffix}.tflite")

    if not tflite_path.exists():
        # Fallback: search current dir
        matches = list(Path(".").rglob("*.tflite"))
        if matches:
            tflite_path = matches[0]
        else:
            raise FileNotFoundError(
                "Export finished but .tflite file not found. "
                "Check the ultralytics export output above."
            )

    print(f"[3/3] Export complete: {tflite_path.resolve()}")
    return tflite_path


def copy_to_assets(tflite_path: Path) -> None:
    """Copy the exported model to the Android assets directory."""
    assets_dir = Path(
        "android_app/app/src/main/assets"
    )
    assets_dir.mkdir(parents=True, exist_ok=True)

    dest = assets_dir / "yolov8n.tflite"
    shutil.copy2(tflite_path, dest)
    print(f"\n✅  Model copied to: {dest.resolve()}")
    print("   You can now build and run the Android app.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export YOLOv8n → TFLite for ADAS Android app")
    parser.add_argument("--int8",   action="store_true", help="Export INT8 quantized model (faster, smaller)")
    parser.add_argument("--imgsz",  type=int, default=320, help="Input image size (default: 320)")
    parser.add_argument("--no-copy", action="store_true", help="Skip auto-copy to assets/ folder")
    args = parser.parse_args()

    tflite_file = export_yolov8_tflite(imgsz=args.imgsz, use_int8=args.int8)

    if not args.no_copy:
        copy_to_assets(tflite_file)
    else:
        print(f"\nModel exported to: {tflite_file.resolve()}")
        print("Copy it manually to:  android_app/app/src/main/assets/yolov8n.tflite")
