import onnx
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "yolo/weights/test_static.onnx"
model = onnx.load(str(INPUT_PATH))
for prop in model.metadata_props:
    if prop.key == "names":
        print("Found classes:", json.loads(prop.value))