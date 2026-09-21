import sys
import os
import argparse
import time

import torch
import numpy as np
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import get_model
from utils.checkpoint import load_model_for_inference


def export_to_onnx(checkpoint_path, output_path, model_name="deeplabv3plus",
                   num_classes=10, image_size=512, opset=17):
    """
    Convert a PyTorch checkpoint to ONNX format.

    ONNX (Open Neural Network Exchange) is a universal format that lets you
    run models on any platform — edge devices, browsers, mobile, cloud — 
    without needing PyTorch installed. Much faster inference too.
    """
    device = torch.device("cpu")  # export always done on CPU for compatibility

    # load trained model
    model = get_model(model_name, n_classes=num_classes)
    model, metadata = load_model_for_inference(checkpoint_path, model, device)
    model.eval()

    # create dummy input matching expected shape [batch, channels, height, width]
    dummy_input = torch.randn(1, 3, image_size, image_size)

    # export to ONNX
    print(f"Exporting {model_name} to ONNX...")
    print(f"  Input shape:  [1, 3, {image_size}, {image_size}]")
    print(f"  Output shape: [1, {num_classes}, {image_size}, {image_size}]")

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,           # store trained weights inside ONNX file
        opset_version=opset,          # ONNX version (17 = latest stable)
        do_constant_folding=True,     # optimize: fold constant ops at export time
        input_names=["image"],        # name the input tensor
        output_names=["segmentation"],  # name the output tensor
        dynamic_axes={                # allow variable batch size at runtime
            "image": {0: "batch_size"},
            "segmentation": {0: "batch_size"},
        },
    )

    # get file size
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n✅ Exported to: {output_path} ({size_mb:.1f} MB)")
    return output_path


def validate_onnx(onnx_path):
    """
    Verify the ONNX file is valid and well-formed.
    This catches export bugs before deployment.
    """
    import onnx
    print("\nValidating ONNX model...")
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    print("✅ ONNX model is valid")

    # print model info
    graph = model.graph
    print(f"  Inputs:  {[i.name for i in graph.input]}")
    print(f"  Outputs: {[o.name for o in graph.output]}")
    print(f"  Nodes:   {len(graph.node)}")


def benchmark(checkpoint_path, onnx_path, model_name="deeplabv3plus",
              num_classes=10, image_size=512, num_runs=20):
    """
    Compare inference speed: PyTorch vs ONNX Runtime.
    ONNX Runtime is typically 2-4x faster than PyTorch.
    """
    import onnxruntime as ort

    dummy = np.random.randn(1, 3, image_size, image_size).astype(np.float32)

    # --- PyTorch benchmark ---
    device = torch.device("cpu")
    model = get_model(model_name, n_classes=num_classes)
    model, _ = load_model_for_inference(checkpoint_path, model, device)
    model.eval()

    torch_input = torch.from_numpy(dummy)
    # warmup
    with torch.no_grad():
        for _ in range(3):
            model(torch_input)

    start = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            model(torch_input)
    pytorch_time = (time.time() - start) / num_runs * 1000  # ms

    # --- ONNX Runtime benchmark ---
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name

    # warmup
    for _ in range(3):
        session.run(None, {input_name: dummy})

    start = time.time()
    for _ in range(num_runs):
        session.run(None, {input_name: dummy})
    onnx_time = (time.time() - start) / num_runs * 1000  # ms

    speedup = pytorch_time / onnx_time

    print(f"\n{'='*45}")
    print(f"{'Benchmark Results':^45}")
    print(f"{'='*45}")
    print(f"  PyTorch:      {pytorch_time:>8.1f} ms/image")
    print(f"  ONNX Runtime: {onnx_time:>8.1f} ms/image")
    print(f"  Speedup:      {speedup:>8.1f}x")
    print(f"{'='*45}")


def run_onnx_prediction(onnx_path, image_path, output_path, image_size=512):
    """
    Run inference using ONNX Runtime instead of PyTorch.
    This proves the exported model works independently.
    """
    import onnxruntime as ort
    import matplotlib.pyplot as plt

    CLASS_NAMES = [
        "Airplane", "Ship", "Storage Tank", "Baseball Diamond",
        "Tennis Court", "Basketball Court", "Ground Track Field",
        "Harbor", "Bridge", "Vehicle"
    ]

    # load and preprocess image (same as training pipeline)
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    tensor = transform(image).unsqueeze(0).numpy()  # [1, 3, H, W] as numpy

    # run ONNX inference
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: tensor})[0]  # [1, 10, H, W]

    # sigmoid + threshold (same as PyTorch predict)
    probs = 1 / (1 + np.exp(-output[0]))  # numpy sigmoid
    mask = probs > 0.5

    # visualize
    image_resized = np.array(image.resize((image_size, image_size)))
    detected = [CLASS_NAMES[i] for i in range(10) if mask[i].any()]
    print(f"Detected: {detected}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(image_resized)
    axes[0].set_title("Original")
    axes[0].axis("off")

    overlay = image_resized.copy()
    colors = [[255,0,0],[0,255,0],[0,0,255],[255,255,0],[255,0,255],
              [0,255,255],[128,0,0],[0,128,0],[0,0,128],[128,128,0]]
    for i in range(10):
        if mask[i].any():
            for c in range(3):
                overlay[:,:,c] = np.where(mask[i],
                    overlay[:,:,c] * 0.5 + colors[i][c] * 0.5,
                    overlay[:,:,c])
    axes[1].imshow(overlay.astype(np.uint8))
    axes[1].set_title(f"ONNX Prediction ({len(detected)} classes)")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Export DeepLab to ONNX")
    parser.add_argument("--checkpoint", required=True, help="path to .pth checkpoint")
    parser.add_argument("--output", default="checkpoints/deeplabv3plus.onnx", help="output .onnx path")
    parser.add_argument("--model", default="deeplabv3plus", help="model name")
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--benchmark", action="store_true", help="run speed benchmark")
    parser.add_argument("--test-image", type=str, help="test image for ONNX prediction")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Step 1: Export
    export_to_onnx(args.checkpoint, args.output, args.model,
                   args.num_classes, args.image_size)

    # Step 2: Validate
    validate_onnx(args.output)

    # Step 3: Benchmark (optional)
    if args.benchmark:
        benchmark(args.checkpoint, args.output, args.model,
                  args.num_classes, args.image_size)

    # Step 4: Test prediction (optional)
    if args.test_image:
        os.makedirs("results/onnx", exist_ok=True)
        basename = os.path.splitext(os.path.basename(args.test_image))[0]
        run_onnx_prediction(args.output, args.test_image,
                            f"results/onnx/{basename}_onnx.png", args.image_size)


if __name__ == "__main__":
    main()