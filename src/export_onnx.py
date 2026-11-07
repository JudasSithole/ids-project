#!/usr/bin/env python3
"""
Export trained MLP model to ONNX format for edge deployment.
Includes preprocessing (StandardScaler) in the pipeline.
"""
from pathlib import Path
import json
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"
MODEL_DIR = ROOT / "models"

print("Loading trained MLP model...", flush=True)
bundle = joblib.load(MODEL_DIR / "neural_network_baseline.joblib")
model = bundle["model"]
scaler = bundle["scaler"]
features = bundle["features"]

# Create pipeline with scaler and model
pipeline = Pipeline([
    ('scaler', scaler),
    ('mlp', model)
])

# Define input type
n_features = len(features)
initial_type = [('float_input', FloatTensorType([None, n_features]))]

print(f"Converting pipeline with {n_features} features to ONNX...", flush=True)
onx = convert_sklearn(pipeline, initial_types=initial_type, target_opset=12)

onnx_path = MODEL_DIR / "neural_network.onnx"
with open(onnx_path, "wb") as f:
    f.write(onx.SerializeToString())

print(f"Saved ONNX model to: {onnx_path}", flush=True)
print(f"Model size: {onnx_path.stat().st_size / 1024:.2f} KB", flush=True)

# Try to quantize
try:
    from onnxruntime.quantization import quantize_dynamic, QuantType
    print("\nApplying dynamic quantization...", flush=True)
    quantized_path = MODEL_DIR / "neural_network_quantized.onnx"
    quantize_dynamic(
        str(onnx_path),
        str(quantized_path),
        weight_type=QuantType.QUInt8
    )
    print(f"Saved quantized ONNX model to: {quantized_path}", flush=True)
    print(f"Quantized model size: {quantized_path.stat().st_size / 1024:.2f} KB", flush=True)
    print(f"Size reduction: {(1 - quantized_path.stat().st_size/onnx_path.stat().st_size)*100:.1f}%", flush=True)
except Exception as e:
    print(f"Quantization failed: {e}", flush=True)
    print("You may need to install: pip install onnx onnxruntime", flush=True)
