# YOLOv11 Skier & Snowboarder Detection

This project utilizes Ultralytics YOLOv11 to detect **Skiers** and **Snowboarders** in alpine environments. The model is optimized for high accuracy and can be used for ski resort surveillance, athlete performance analysis, and automated highlight generation.

## 🚀 Model Performance

After training for 100 epochs, the model achieved the following results on the validation set:

- **mAP50 (Overall)**: 0.952
- **Skier mAP50**: 0.964
- **Snowboarder mAP50**: 0.938

Key evaluation plots (e.g., `results.png`, `BoxF1_curve.png`) are included in the results package.

## 📂 File Description

- `weights/best.pt`: PyTorch weights (recommended for Python development).
- `weights/best.onnx`: ONNX format model (ideal for web, mobile, or edge deployment).
- `data.yaml`: Dataset configuration defining category mappings.
- `results/`: Complete training logs, including confusion matrices and loss curves.

## 💻 How to Use

### 1. Requirements
```bash
pip install ultralytics
```

### 2. Python Inference
```python
from ultralytics import YOLO

# Load the best weights
model = YOLO('weights/best.pt')

# Run inference
results = model.predict(source='path/to/your/video_or_image.jpg', conf=0.5, save=True)

# View results
results[0].show()
```

## 🌟 Key Advantages
- **Precise Classification**: By distinguishing between skiers and snowboarders, the model enables specialized trajectory analysis and predictive blind-spot estimation based on equipment-specific movement patterns.
- **Optimized for Proximity**: Specifically tuned for mid-to-close range detection, significantly reducing false alarms and providing actionable collision warnings for safety systems.


## ⚠️ Limitations & Future Work
- **Epoch Selection**: Training was intentionally capped at 100 epochs to prevent overfitting. While this ensures better generalization to new ski resorts, the absolute peak accuracy might be slightly lower than a fully converged model.
- **Data Scale**: The current performance is based on a focused dataset. Performance may vary in extreme environmental conditions (e.g., heavy snow or low-light night skiing).


## 🛠 Deployment Advice
- **Confidence Threshold**: A threshold between `0.4 - 0.6` is recommended to balance precision and recall, based on the `BoxF1_curve.png`.
- **ONNX Deployment**: For non-Python environments (C++, JavaScript, etc.), please use the exported `best.onnx` file.

## ⚖️ License
[MIT License]
```
