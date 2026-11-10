#!/usr/bin/env python3
"""Check the existing course inputs and optional local DINO/probe assets."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys

MODEL_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"
PROBE_NAME = "dinov3_vitl16_web_bn_posw4.pt"


def project_root():
    value = os.environ.get("EC523_PROJECT_DIR")
    if not value:
        raise RuntimeError("First run: source scripts/activate_scc.sh")
    root = Path(value).resolve()
    if not (root / "scripts/train_refiner.py").is_file():
        raise RuntimeError(f"This is not the road-segmentation repository: {root}")
    return root


def encoder_snapshot():
    """Use the same cache location/main reference as the project's HF loader."""
    home = os.environ.get("HF_HOME")
    if not home:
        raise RuntimeError("HF_HOME is unset; source scripts/activate_scc.sh first")
    cache = Path(home) / "hub"
    os.environ["HF_HUB_CACHE"] = str(cache)
    model_cache = cache / ("models--" + MODEL_ID.replace("/", "--"))
    reference = model_cache / "refs/main"
    if not reference.is_file():
        raise RuntimeError("DINOv3 is not cached. Run download_encoder.py first.")
    revision = reference.read_text().strip()
    if not revision or any(c not in "0123456789abcdef" for c in revision):
        raise RuntimeError("The cached DINOv3 main reference is invalid")
    snapshot = model_cache / "snapshots" / revision
    for name in ("config.json", "preprocessor_config.json"):
        with (snapshot / name).open() as stream:
            json.load(stream)
    index = snapshot / "model.safetensors.index.json"
    if index.is_file():
        names = set(json.loads(index.read_text())["weight_map"].values())
        if not names:
            raise RuntimeError("DINOv3 weight index is empty")
    else:
        names = {"model.safetensors"}
    for name in names:
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise RuntimeError("Invalid weight filename in the model index")
        path = snapshot / name
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing/empty DINOv3 weights: {path}")
    return snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", action="store_true", help="Check cached model files")
    parser.add_argument("--probe", action="store_true", help="Validate the trained probe checkpoint")
    parser.add_argument("--gpu", action="store_true", help="Load and forward the encoder on an allocated GPU")
    args = parser.parse_args()
    root = project_root()
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[variable] = "1"
    os.environ["HF_HUB_CACHE"] = str(Path(os.environ["HF_HOME"]) / "hub")
    sys.path[:0] = [str(root), str(root / "src")]

    import torch
    import torchvision
    import transformers
    import scipy
    import skimage
    torch.set_num_threads(1)
    from scripts.train_linear_probe import official_metrics
    from looproad.data.road_dataset import RoadDataset

    print(f"Repository: {root}")
    for package in ("torch", "torchvision", "transformers", "numpy", "huggingface_hub"):
        print(f"{package}: {importlib.metadata.version(package)}")
    starter = Path(os.environ.get("EC523_STARTER_DIR", ""))
    for name in ("metrics.py", "common.py"):
        if not (starter / name).is_file():
            raise RuntimeError(f"Missing {name} in EC523_STARTER_DIR={starter}")
    metrics = official_metrics()
    for name in ("Counts", "image_counts", "summarize"):
        if not hasattr(metrics, name):
            raise RuntimeError(f"The course metrics module lacks {name}")
    print(f"Course starter: {starter}")
    for split in ("Train", "Val"):
        for domain in ("Urban", "Rural"):
            dataset = RoadDataset(split, domain)
            if not len(dataset):
                raise RuntimeError(
                    f"No images in {os.environ.get('EC523_DATA_DIR')}/{split}/{domain}/images_png. "
                    "Check the course dataset path and access permissions."
                )
            dataset[0]  # Validate one real image/mask; constructor checks all paired masks.
            print(f"Dataset {split}/{domain}: {len(dataset)} pairs; first pair decoded OK")

    if args.encoder or args.gpu:
        print(f"Encoder cache: {encoder_snapshot()}")
        transformers.AutoConfig.from_pretrained(MODEL_ID, local_files_only=True)
        transformers.AutoImageProcessor.from_pretrained(MODEL_ID, local_files_only=True)
    head = None
    if args.probe:
        from looproad.models.linear_probe import LinearProbe
        path = root / "runs/linear_probe/checkpoints" / PROBE_NAME
        if not path.is_file():
            raise RuntimeError(f"Train the probe first; checkpoint missing: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if (checkpoint.get("encoder") != "dinov3_vitl16_web"
                or checkpoint.get("head") != "bn_linear"
                or float(checkpoint.get("pos_weight", -1)) != 4):
            raise RuntimeError("Probe must be dinov3_vitl16_web, bn_linear, pos_weight=4")
        head = LinearProbe(1024, head_type="bn_linear").eval()
        head.load_state_dict(checkpoint["state_dict"], strict=True)
        if any(not torch.isfinite(tensor).all() for tensor in head.state_dict().values()):
            raise RuntimeError("Probe checkpoint contains nonfinite values")
        print(f"Probe checkpoint: {path}; metadata and weights OK")

    if args.gpu:
        if not os.environ.get("JOB_ID") or not torch.cuda.is_available():
            raise RuntimeError("--gpu requires a running SCC GPU job; submit the .qsub file")
        from looproad.encoders.resnet import build_encoder
        encoder = build_encoder("dinov3_vitl16_web").to("cuda").eval()
        sample = RoadDataset("Train", "Urban")[0][0]
        with torch.no_grad():
            features = encoder(torch.from_numpy(sample).unsqueeze(0).to("cuda"))
            if not torch.isfinite(features).all():
                raise RuntimeError("Encoder forward produced nonfinite values")
            if head is not None:
                logits = head.to("cuda")(features)
                if logits.shape != (1, 1, 1024, 1024) or not torch.isfinite(logits).all():
                    raise RuntimeError("Probe forward failed shape/finiteness validation")
        print(f"GPU: {torch.cuda.get_device_name()}; real-image forward OK")
    print("INPUT CHECK: PASS")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"INPUT CHECK FAILED: {error}", file=sys.stderr)
        sys.exit(1)
