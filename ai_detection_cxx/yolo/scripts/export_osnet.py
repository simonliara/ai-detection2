import torch
import torch.nn as nn

def _clean_state_dict(sd):
    # common checkpoint formats: {"state_dict": ...} / {"model": ...} / raw sd
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]

    out = {}
    for k, v in sd.items():
        # strip common prefixes
        for prefix in ("module.", "model."):
            if k.startswith(prefix):
                k = k[len(prefix):]
        out[k] = v
    return out

def export_osnet_dynamic(pth_path, output_onnx):
    import torchreid

    # Build OSNet backbone
    model = torchreid.models.build_model(
        name="osnet_x0_25",
        num_classes=1000,      # doesn't really matter if we remove classifier
        pretrained=False
    )

    # Remove classifier so output is embeddings/features
    if hasattr(model, "classifier"):
        model.classifier = nn.Identity()

    ckpt = torch.load(pth_path, map_location="cpu")
    sd = _clean_state_dict(ckpt)

    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    model.eval()

    dummy = torch.randn(1, 3, 256, 128)

    torch.onnx.export(
        model,
        dummy,
        output_onnx,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=["images"],      # Keep as 'images'
        output_names=["output"],      # CHANGE THIS from 'features' to 'output'
        dynamic_axes={
            "images": {0: "batch"},
            "output": {0: "batch"},   # Match the name here too
        },
    )
    print(f"Exported {output_onnx} (dynamic batch)")

if __name__ == "__main__":
    export_osnet_dynamic(
        "botsort/weights/osnet_x0_25_imagenet.pth",
        "botsort/weights/osnet_dynamic.onnx",
    )