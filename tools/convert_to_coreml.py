"""
Convert mobilenetv2.pt → MobileGaze.mlpackage via torch.jit.trace.

coremltools 7+ dropped ONNX support; convert directly from PyTorch instead.

Usage:
    python convert_to_coreml.py
"""

import torch
import coremltools as ct
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import data_config
from utils.helpers import get_model

WEIGHT_PATH = "weights/mobilenetv2.pt"
OUTPUT_PATH = "../GazeTrackingApp/MobileGaze.mlpackage"

cfg = data_config["gaze360"]   # bins=90, binwidth=4, angle=180

device = torch.device("cpu")
model  = get_model("mobilenetv2", cfg["bins"], inference_mode=True)
model.load_state_dict(torch.load(WEIGHT_PATH, map_location=device))
model.eval()
print("Model loaded.")

# Trace with a dummy 448x448 input (matches training transform)
dummy = torch.randn(1, 3, 448, 448)
traced = torch.jit.trace(model, dummy)
print("Traced.")

print(f"Converting to Core ML ...")
mlmodel = ct.convert(
    traced,
    inputs=[ct.TensorType(name="input", shape=dummy.shape)],
    minimum_deployment_target=ct.target.iOS15,
    compute_units=ct.ComputeUnit.ALL,
)

mlmodel.author            = "Gaze Tracking Research Team"
mlmodel.short_description = "MobileNetV2 gaze estimator — yaw/pitch logits (90 bins, gaze360)"
mlmodel.input_description["input"]   = "448x448 float32 tensor, ImageNet-normalized (C,H,W)"

mlmodel.save(OUTPUT_PATH)
print(f"Saved → {OUTPUT_PATH}")

# Print actual output names (needed for Swift inference code)
spec = mlmodel.get_spec()
print("\nModel spec:")
for inp in spec.description.input:
    print(f"  input : {inp.name}")
for out in spec.description.output:
    print(f"  output: {out.name}  ← use this name in Swift")
print("Done.")
