"""Train a two-class YOLOv5u model and optionally export a TensorRT engine.

Paths are resolved from this file, not the caller's current directory. Dataset
preparation can run without installing Ultralytics, CUDA, or TensorRT.
"""

import argparse
from pathlib import Path
import shutil
import subprocess

from prepare_dataset import prepare_dataset

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "D2C2/datasets/RMCOCO2")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "D2C2/train/yolo_output")
    parser.add_argument("--weights", default="yolov5nu.pt",
                        help="Ultralytics anchor-free YOLOv5u checkpoint")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--export-engine", action="store_true")
    parser.add_argument("--trtexec", default="trtexec")
    parser.add_argument("--engine-path", type=Path,
                        default=ROOT / "models/yolov5n_d2c2_fp16.engine")
    args = parser.parse_args()
    dataset = prepare_dataset(args.data_dir, args.output_dir, args.val_fraction, args.seed)
    print(f"Dataset configuration: {dataset}")
    if args.prepare_only:
        return
    from ultralytics import YOLO

    model = YOLO(args.weights)
    model.train(data=str(dataset), epochs=args.epochs, imgsz=416, rect=True,
                batch=args.batch, device=args.device, workers=4, lr0=1e-4,
                patience=15, save=True, seed=args.seed,
                project=str(args.output_dir.resolve()), name="train", exist_ok=False)
    # Use this run's checkpoint, never a recursive search for an older best.pt.
    best_pt = Path(model.trainer.best)
    if not best_pt.is_file():
        raise FileNotFoundError(f"Training did not produce a checkpoint: {best_pt}")
    export_model = YOLO(str(best_pt))
    onnx_path = Path(export_model.export(format="onnx", imgsz=(224, 416),
                                        batch=1, dynamic=False, half=False,
                                        nms=False, device=args.device, simplify=False))
    print(f"ONNX model: {onnx_path}")
    if args.export_engine:
        executable = shutil.which(args.trtexec)
        if not executable:
            raise FileNotFoundError(f"TensorRT executable not found: {args.trtexec}")
        args.engine_path.parent.mkdir(parents=True, exist_ok=True)
        if args.engine_path.exists():
            raise FileExistsError(f"Refusing to overwrite an existing engine: {args.engine_path}")
        subprocess.run([executable, f"--onnx={onnx_path}",
                        f"--saveEngine={args.engine_path}", "--fp16"], check=True)
        print(f"TensorRT engine: {args.engine_path}")


if __name__ == "__main__":
    main()
