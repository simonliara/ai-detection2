from ultralytics import YOLO

model = YOLO("/home/simonli/ObjectDetection/yolo/weights/yolo11s_static.fp16.engine")
results = model("https://ultralytics.com/images/bus.jpg", save=True)
print(f"Device used: {results[0].probs if hasattr(results[0], 'probs') else 'GPU'}")
print("Success! Detections made.")