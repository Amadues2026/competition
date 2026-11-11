# Third-party components and asset provenance

Repository-authored software follows the Apache-2.0 declaration already present in the ROS package manifests. That declaration does not grant rights to third-party code, weights, data, or competition documents. This inventory records dependencies and known gaps; it is not a claim that all assets have been independently audited.

| Component / asset | Source and status |
| --- | --- |
| ROS2 / rclcpp / ament | [ROS2](https://github.com/ros2), distributed under their respective package licenses. |
| OpenCV | [OpenCV](https://github.com/opencv/opencv); follow the license for the installed version. |
| yaml-cpp | [yaml-cpp](https://github.com/jbeder/yaml-cpp), MIT. |
| CUDA and TensorRT | NVIDIA SDKs, subject to NVIDIA's applicable terms. Not relicensed by this repository. |
| Ultralytics and pretrained checkpoints | [Ultralytics](https://github.com/ultralytics/ultralytics), with upstream licensing requirements. Training code and resulting weights require a separate license review for intended distribution/use. |
| Existing `models/yolov5n_d2c2_fp16.engine` | Preserved from the repository. Exact source checkpoint, training data permissions, conversion environment, and redistribution terms are not documented here. Do not assume Apache-2.0 covers this binary. |
| `D2C2/datasets/` | Not distributed. Supply data you are authorized to use and document its source. |
| Competition rules and simulator manual | Preserved under `docs/reference/`; external reference documents retain their original terms. |
| GoogleTest | Used by ament-based unit tests under its upstream license. |

When adding an asset, record its upstream URL, exact version or hash, license, local modifications, and any required notices. The maintainer should resolve the existing engine and dataset provenance before presenting them as freely redistributable assets.
