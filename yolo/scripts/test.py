from ultralytics import YOLO
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
model_path = repo_root / "yolo/weights/yolo11s_static.fp16.engine"
model = YOLO(str(model_path))
results = model("https://ultralytics.com/images/bus.jpg", save=True)
print(f"Device used: {results[0].probs if hasattr(results[0], 'probs') else 'GPU'}")
print("Success! Detections made.")