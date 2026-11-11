# D2C2 model pipeline

Dataset preparation, training, ONNX export, and TensorRT conversion for the red/blue armor detector. Python 3.10+ is required. Data, pretrained weights, and training output are not included in the repository.

## Dataset layout

```text
D2C2/datasets/RMCOCO2/
  annotations/annotations.json   COCO images/categories/annotations
  images/                       Images referenced by relative file_name
  labels/                       Generated YOLO labels
```

Required category names are `red` and `blue`, mapped to class 0 and class 1. Image names may contain subdirectories. Labels are written beside the corresponding image tree under `labels/`; images without recognized boxes receive empty label files. Back up any independently maintained labels before regenerating them.

## Prepare and inspect a split without GPU dependencies

```bash
python D2C2/train/train_yolov5.py --prepare-only
```

Defaults resolve relative to the script location, so the command may be invoked from another working directory. To use external data or outputs:

```bash
python D2C2/train/train_yolov5.py --prepare-only \
  --data-dir /absolute/path/to/RMCOCO2 --output-dir /tmp/tdt-dataset \
  --val-fraction 0.2 --seed 42
```

The tool writes labels, `train.txt`, `val.txt`, and `dataset.yaml`. The two image lists are disjoint and deterministic for the same dataset/seed. This is an image-level split, not a recording-level split: adjacent frames or duplicate content can still leak information between subsets. Prepare session-separated evaluation data before claiming generalization.

## Train and export ONNX

Use an isolated environment outside the repository for dependencies:

```bash
python -m pip install -r D2C2/train/requirements_train.txt
python D2C2/train/train_yolov5.py --device 0 --epochs 100 --batch 8
```

The training entry point now selects `yolov5nu.pt` explicitly: the Ultralytics YOLOv5u variant, not a claim that legacy YOLOv5 checkpoints are identical. Training uses a scalar size of 416 with rectangular batches; export uses a static `(224, 416)` input and batch 1. Export has no integrated NMS and uses float I/O. The runtime expects `images`, `output0`, and two class scores after four box coordinates. See [Ultralytics YOLOv5u](https://docs.ultralytics.com/models/yolov5/) and [export documentation](https://docs.ultralytics.com/modes/export/).

Each run uses a separate output directory. The script exports the checkpoint selected by that run's trainer rather than searching for an arbitrary older `best.pt`.

## Convert to TensorRT on the deployment machine

```bash
bash D2C2/scripts/convert_to_tensorrt.sh \
  /absolute/path/to/best.onnx /absolute/path/to/new-model.engine
```

Set `TRTEXEC=/path/to/trtexec` if it is not on PATH. The converter passes `--fp16` to enable FP16 kernels, preserves float network I/O, and refuses an existing output path. FP16-enabled does not mean every layer necessarily runs at FP16 precision.

Alternatively append `--export-engine --engine-path /path/to/new.engine` to the training command. `trtexec` failures propagate as errors. Existing engine files are not overwritten. Serialized engines are environment-specific: validate TensorRT/CUDA/GPU compatibility and compare predictions before deployment.

## Evaluation status

Earlier notes reported ~2 ms latency and mAP50 0.90. No matching held-out evaluation data, hardware description, or reproducible report is checked in. The previous script used the same directory for training and validation. Those historical figures must not be treated as verified test-set results. This maintenance update does not retrain or validate the existing engine.
