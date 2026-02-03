import onnx
import json
import sys
from pathlib import Path

DYNAMIC = False
SUFFIX = "dynamic" if DYNAMIC else "static"

PROJECT_ROOT = Path(__file__).resolve().parents[2] 
ONNX_PATH = PROJECT_ROOT / f"yolo/weights/test_{SUFFIX}.onnx"

def extract_metadata():
    path = Path(ONNX_PATH)
    if not path.exists():
        print(f"[ERROR] ONNX file not found at: {path}")
        print("Did you run the export script first?")
        sys.exit(1)

    print(f"Loading {path}...")
    model = onnx.load(str(path))

    labels_json = None
    for prop in model.metadata_props:
        if prop.key == "names":
            labels_json = prop.value
            break

    if labels_json:
        output_path = path.with_suffix(".txt")
        try:
            data = json.loads(labels_json)
            
            names_list = []
            if isinstance(data, dict):
                sorted_keys = sorted(data.keys(), key=lambda x: int(x))
                names_list = [data[k] for k in sorted_keys]
            elif isinstance(data, list):
                names_list = data
            else:
                print(f"[ERROR] Unexpected format for 'names': {type(data)}")
                return

            with open(output_path, "w") as f:
                for name in names_list:
                    f.write(f"{name}\n")
                
            print(f"[OK] Extracted {len(names_list)} labels to: {output_path}")
            
        except json.JSONDecodeError:
            print("[ERROR] Metadata 'names' found but invalid JSON.")
    else:
        print("[WARN] No 'names' metadata found in ONNX.")

if __name__ == "__main__":
    extract_metadata()