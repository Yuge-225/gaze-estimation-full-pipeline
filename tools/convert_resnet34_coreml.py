"""
Convert resnet34.pt → ResNet34Gaze.mlpackage for CoreML / iOS.

Run from the gaze-estimation directory:
    python convert_resnet34_coreml.py

Requires: torch, torchvision, coremltools
    pip install coremltools
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import coremltools as ct
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.helpers import get_model

# ── Config ────────────────────────────────────────────────────────────────────
WEIGHTS   = "weights/resnet34.pt"
OUT_PATH  = "weights/ResNet34Gaze.mlpackage"
BINS      = 90
INPUT_SIZE = 448
DEVICE    = torch.device("cpu")   # CoreML conversion must use CPU

# ── Load model ────────────────────────────────────────────────────────────────
print("Loading ResNet34 model …")
model = get_model("resnet34", BINS, inference_mode=True)
model.load_state_dict(torch.load(WEIGHTS, map_location=DEVICE))
model.eval()

# ── Wrap to return a single output dict for CoreML compatibility ──────────────
# CoreML handles tuple outputs but naming can be unpredictable;
# wrapping in a dict gives explicit names.
class ResNet34Wrapper(torch.nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base

    def forward(self, x):
        yaw, pitch = self.base(x)
        return yaw, pitch

wrapped = ResNet34Wrapper(model).eval()

# ── Trace ─────────────────────────────────────────────────────────────────────
print("Tracing model …")
dummy = torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE)
with torch.no_grad():
    traced = torch.jit.trace(wrapped, dummy)

# ── Convert to CoreML ─────────────────────────────────────────────────────────
print("Converting to CoreML …")
mlmodel = ct.convert(
    traced,
    inputs=[ct.TensorType(name="input", shape=dummy.shape)],
    outputs=[
        ct.TensorType(name="yaw_logits"),
        ct.TensorType(name="pitch_logits"),
    ],
    minimum_deployment_target=ct.target.iOS16,
    compute_precision=ct.precision.FLOAT32,
)

# ── Save ──────────────────────────────────────────────────────────────────────
mlmodel.save(OUT_PATH)
print(f"Saved → {OUT_PATH}")

# ── Verify output names ───────────────────────────────────────────────────────
print("\nModel spec outputs:")
for name, desc in mlmodel.output_description._fd_spec.items():
    print(f"  {name}")
