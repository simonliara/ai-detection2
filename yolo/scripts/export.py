from ultralytics import YOLO
from pathlib import Path
import shutil

DYNAMIC = False

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "yolo/weights/yolo11s.pt"

input_directory = INPUT_PATH.parent
input_name = INPUT_PATH.stem
suffix = "dynamic" if DYNAMIC else "static"
desired = input_directory / f"{input_name}_{suffix}.onnx"

model = YOLO(str(INPUT_PATH))

exported = Path(
    model.export(
        format="onnx",
        imgsz=640,
        dynamic=DYNAMIC,
        simplify=True,
    )
)

shutil.move(exported, desired)

print("Exported:", exported)
print("Renamed to:", desired)
