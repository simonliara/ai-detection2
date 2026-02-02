from ultralytics import YOLO
from pathlib import Path
import shutil
import json
import onnx 

DYNAMIC = False

CLASS_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "yolo/weights/yolo11s.pt"

input_directory = INPUT_PATH.parent
input_name = INPUT_PATH.stem
suffix = "dynamic" if DYNAMIC else "static"
desired = input_directory / f"{input_name}_{suffix}.onnx"

model = YOLO(INPUT_PATH)

exported_path_str = model.export(
    format="onnx",
    imgsz=640,
    dynamic=DYNAMIC,
    simplify=True
)
exported = Path(exported_path_str)

print(f"Loading {exported} to inject metadata...")
model_onnx = onnx.load(str(exported))

names_dict = {i: name for i, name in enumerate(CLASS_NAMES)}
meta_value = json.dumps(names_dict)

meta_found = False
for prop in model_onnx.metadata_props:
    if prop.key == "names":
        prop.value = meta_value
        meta_found = True
        break

if not meta_found:
    meta = model_onnx.metadata_props.add()
    meta.key = "names"
    meta.value = meta_value

onnx.save(model_onnx, str(exported))
print("Metadata injected successfully.")

shutil.move(str(exported), str(desired))
print("Renamed to:", desired)