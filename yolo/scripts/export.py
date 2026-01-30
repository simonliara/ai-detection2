from ultralytics import YOLO
from pathlib import Path
import shutil

DYNAMIC = False
INPUT_PATH = "/home/simonli/ObjectDetection/yolo/weights/t2.pt"

input_directory = Path(INPUT_PATH).parent
input_name = Path(INPUT_PATH).stem
suffix = "dynamic" if DYNAMIC else "static"
desired = input_directory / f"{input_name}_{suffix}.onnx"

model = YOLO(INPUT_PATH)

exported = Path(model.export(
    format="onnx",
    imgsz=640,
    dynamic=DYNAMIC,
    simplify=True
))

shutil.move(str(exported), str(desired))
print("Exported:", exported)
print("Renamed to:", desired)
