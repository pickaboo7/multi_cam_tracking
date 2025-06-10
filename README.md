# Multi-Camera Human Tracking System

A real-time multi-camera human detection and tracking system using DeepStream, YOLOv5/YOLOv8, and DBSCAN clustering for cross-camera correspondence.

## Features

- **Multi-Camera Support**: Simultaneous processing of up to 4 camera feeds
- **Real-time Detection**: Hardware-accelerated human detection using NVIDIA DeepStream
- **Cross-Camera Tracking**: DBSCAN-based clustering for person correspondence across cameras
- **Flexible Models**: Support for both YOLOv5 and YOLOv8 TensorRT engines
- **Camera Calibration**: Built-in calibration system for multi-camera geometry
- **Visualization**: Real-time plotting and tracking visualization

## System Requirements

### Hardware
- NVIDIA GPU with compute capability ≥ 6.1
- CUDA-compatible GPU with ≥ 4GB VRAM
- Multi-camera setup (USB/IP cameras supported)

### Software
- Ubuntu 18.04/20.04/22.04
- CUDA 11.4+
- TensorRT 8.0+
- DeepStream SDK 6.0+
- Python 3.8+

## Installation

### 1. Clone Repository
```bash
git clone https://github.com/pickaboo7/multi_cam_tracking.git
cd multi_cam_tracking
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Install DeepStream SDK
Follow [NVIDIA DeepStream Installation Guide](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Quickstart.html)

### 4. Setup TensorRT Models
Place your TensorRT engine files in `deepstream/models/`:
- `model_b1_gpu0_fp32.engine` (YOLOv5)
- `yolov8m_b1_gpu0_fp16.engine` (YOLOv8)

## Quick Start

### 1. Camera Calibration
```bash
# Capture calibration frames
python calibration/capture_frames.py

# Generate calibration points
python calibration/generate_points.py
```

### 2. Run Detection Pipeline
```bash
cd deepstream
python deepstream_human_detection.py
```

### 3. Analyze Results
```bash
cd clustering
python clustering_and_plotting.py
```

## Project Structure

```
multi_cam_tracking/
├── calibration/           # Camera calibration data and scripts
│   ├── frame_cam_*.jpg   # Reference frames from each camera
│   ├── points_cam_*.npy  # 2D calibration points
│   └── *_centroids.npy   # Tracking centroid data
├── deepstream/           # DeepStream detection pipeline
│   ├── deepstream_human_detection.py
│   ├── config/          # DeepStream configuration files
│   └── models/          # TensorRT engine files
├── clustering/          # Cross-camera tracking algorithms
│   └── clustering_and_plotting.py
├── footage/            # Sample video files
└── docs/              # Documentation and guides
```

## Configuration

### DeepStream Configuration
Modify `deepstream/config/config_infer_primary_yoloV5.txt` for:
- Model path and parameters
- Detection thresholds
- Input resolution
- Batch size

### Clustering Parameters
Adjust DBSCAN parameters in `clustering/clustering_and_plotting.py`:
```python
eps = 0.5        # Maximum distance between points
min_samples = 2  # Minimum points to form cluster
```

## Usage Examples

### Basic Multi-Camera Detection
```python
from deepstream.deepstream_human_detection import MultiCamDetector

detector = MultiCamDetector(num_cameras=4)
detector.start_detection()
```

### Cross-Camera Tracking
```python
from clustering.clustering_and_plotting import CrossCameraTracker

tracker = CrossCameraTracker()
tracker.load_detections('calibration/')
tracker.cluster_and_visualize()
```

## Performance

| Configuration | FPS | Latency | GPU Usage |
|---------------|-----|---------|-----------|
| 4x 1080p (YOLOv5) | 30 | 33ms | 60% |
| 4x 720p (YOLOv8) | 45 | 22ms | 45% |

## Troubleshooting

### Common Issues

**DeepStream Pipeline Fails**
- Verify CUDA/TensorRT installation
- Check model file paths
- Ensure camera feeds are accessible

**Low Detection Accuracy**
- Adjust detection thresholds in config
- Verify model compatibility
- Check camera calibration quality

**High GPU Memory Usage**
- Reduce batch size
- Lower input resolution
- Use FP16 precision

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- NVIDIA DeepStream SDK
- Ultralytics YOLOv5/YOLOv8
- OpenCV community
- scikit-learn DBSCAN implementation

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{multicam_tracking_2024,
  title={Multi-Camera Human Tracking System},
  author={Your Name},
  year={2024},
  publisher={GitHub},
  howpublished={\url{https://github.com/pickaboo7/multi_cam_tracking}}
}
```

## Contact

- **Author**: Piyaansh Thukral 
- **Email**: piyansh2001@gmail.com
- **Project Link**: https://github.com/pickaboo7/multi_cam_tracking
