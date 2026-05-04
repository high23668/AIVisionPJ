"""
DINOv3 Demo Script
- Feature extraction, zero-shot k-NN recognition, feature visualization
- Models: ViT-B (86M), ViT-L (300M) from HuggingFace
"""
import argparse
import time
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from transformers import AutoModel, AutoImageProcessor


# DINOv3 (Gated repo - requires HuggingFace login: huggingface-cli login)
DINOV3_MODELS = {
    "vitb": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "vitl": "facebook/dinov3-vitl16-pretrain-lvd1689m",
}

# DINOv2 fallback (open access - no login required)
DINOV2_FALLBACK_MODELS = {
    "vitb": "facebook/dinov2-base",
    "vitl": "facebook/dinov2-large",
    "vits": "facebook/dinov2-small",
}


def load_model(variant="vitb", device="cuda", use_dinov2_fallback=False):
    """Load DINOv3 (or DINOv2 fallback) model and processor from HuggingFace.

    DINOv3 requires HuggingFace login: run `huggingface-cli login` first,
    then accept terms at https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m

    If DINOv3 is not accessible, set use_dinov2_fallback=True to use DINOv2.
    DINOv2 has the same architecture but was trained on fewer images (142M vs 1.7B).
    """
    models_dict = DINOV2_FALLBACK_MODELS if use_dinov2_fallback else DINOV3_MODELS
    model_name = "DINOv2" if use_dinov2_fallback else "DINOv3"
    model_id = models_dict[variant]
    print(f"Loading {model_name} {variant}: {model_id}")

    if not use_dinov2_fallback:
        import huggingface_hub
        try:
            huggingface_hub.whoami()
        except Exception:
            print("  WARNING: Not logged in to HuggingFace. Run `huggingface-cli login` for DINOv3.")
            print("  Falling back to DINOv2...")
            return load_model(variant, device, use_dinov2_fallback=True)

    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
    model = model.to(device).eval()
    print(f"  OK - {model_name} {variant} loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
    return model, processor


def extract_features(model, processor, image, device="cuda"):
    """Extract CLS token + patch features from an image."""
    if isinstance(image, (str, Path)):
        image = Image.open(image).convert("RGB")

    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # CLS token: global feature
    cls_feature = outputs.last_hidden_state[:, 0, :]  # (1, hidden_dim)
    # Patch features: local spatial features
    patch_features = outputs.last_hidden_state[:, 1:, :]  # (1, num_patches, hidden_dim)

    return cls_feature.cpu().float(), patch_features.cpu().float()


def benchmark_feature_extraction(model, processor, image_path, device="cuda", n_runs=10):
    """Benchmark feature extraction speed."""
    image = Image.open(image_path).convert("RGB") if isinstance(image_path, (str, Path)) else image_path

    # Warmup
    extract_features(model, processor, image, device)
    torch.cuda.synchronize() if device == "cuda" else None

    times = []
    for _ in range(n_runs):
        torch.cuda.synchronize() if device == "cuda" else None
        t0 = time.perf_counter()
        extract_features(model, processor, image, device)
        torch.cuda.synchronize() if device == "cuda" else None
        times.append((time.perf_counter() - t0) * 1000)

    avg_ms = np.mean(times)
    fps = 1000 / avg_ms
    vram_mb = torch.cuda.memory_allocated() / 1024**2 if device == "cuda" else 0
    return avg_ms, fps, vram_mb


def knn_recognition(query_features, gallery_features, gallery_labels, k=5):
    """Zero-shot k-NN classification using cosine similarity."""
    query = torch.nn.functional.normalize(query_features, dim=-1)
    gallery = torch.nn.functional.normalize(gallery_features, dim=-1)

    # Cosine similarity matrix
    sims = torch.mm(query, gallery.T)  # (n_query, n_gallery)
    topk_vals, topk_idx = sims.topk(k, dim=-1)

    predictions = []
    for i in range(len(query)):
        top_labels = [gallery_labels[j] for j in topk_idx[i].tolist()]
        top_scores = topk_vals[i].tolist()
        # Majority vote
        from collections import Counter
        pred_label = Counter(top_labels).most_common(1)[0][0]
        predictions.append({
            "prediction": pred_label,
            "top_labels": top_labels,
            "top_scores": top_scores,
        })
    return predictions


def visualize_patch_features(model, processor, image, save_path="patch_features.png", device="cuda"):
    """Visualize patch-level features via PCA."""
    from sklearn.decomposition import PCA

    if isinstance(image, (str, Path)):
        img = Image.open(image).convert("RGB")
    else:
        img = image

    _, patch_feats = extract_features(model, processor, img, device)
    # patch_feats: (1, num_patches, hidden_dim)
    patches = patch_feats[0].numpy()  # (num_patches, hidden_dim)

    # PCA to 3 components for RGB visualization
    pca = PCA(n_components=3)
    pca_feats = pca.fit_transform(patches)  # (num_patches, 3)
    pca_feats = (pca_feats - pca_feats.min()) / (pca_feats.max() - pca_feats.min())

    # Reshape to spatial grid (handle non-square patch grids)
    n_patches = patches.shape[0]
    # Find closest w/h factors: e.g. 200 patches → 10×20 or 14×14 with padding
    h = int(n_patches**0.5)
    while h > 1 and n_patches % h != 0:
        h -= 1
    w = n_patches // h
    # Crop to h*w if n_patches doesn't factor evenly
    pca_img = pca_feats[:h * w].reshape(h, w, 3)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(img)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    axes[1].imshow(pca_img)
    axes[1].set_title("DINOv3 Patch Features (PCA)")
    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved feature visualization: {save_path}")


def camera_snapshot(model, processor, source=6, save_dir="results", device="cuda", variant="vitb"):
    """Capture one frame from camera, run DINOv3 PCA visualization, display result."""
    import cv2

    print(f"Capturing from camera source={source}...")
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera source={source}")
        return

    # Discard first few frames while camera sensor stabilizes
    for _ in range(30):
        cap.read()
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("ERROR: Failed to capture frame")
        return

    # Save raw capture
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    raw_path = f"{save_dir}/camera_capture.jpg"
    cv2.imwrite(raw_path, frame)
    print(f"  Captured frame: {frame.shape[1]}x{frame.shape[0]}")

    # BGR -> RGB for PIL
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    save_path = f"{save_dir}/dinov3_{variant}_camera_features.png"
    print("  Running DINOv3 feature extraction + PCA...")
    t0 = time.perf_counter()
    visualize_patch_features(model, processor, pil_img, save_path=save_path, device=device)
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"  Done in {elapsed:.0f}ms")
    print(f"  Result saved: {save_path}")

    # Display result
    result_img = cv2.imread(save_path)
    if result_img is not None:
        cv2.imshow("DINOv3 Patch Features (PCA) - press any key to close", result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def capture_frame(source):
    """Capture a single stable frame from camera."""
    import cv2
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera source={source}")
        return None, None
    for _ in range(30):
        cap.read()
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("ERROR: Failed to capture frame")
        return None, None
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb), frame


def register_object(model, processor, label, source=6, gallery_path="results/gallery.npz", device="cuda"):
    """Capture object from camera and register its DINOv3 feature."""
    import cv2
    input(f"  カメラに「{label}」を向けてEnterを押してください...")
    pil_img, frame = capture_frame(source)
    if pil_img is None:
        return

    cls_feat, _ = extract_features(model, processor, pil_img, device)
    feat_vec = torch.nn.functional.normalize(cls_feat, dim=-1).squeeze(0).numpy()

    # Load existing gallery or create new
    gallery = {}
    if Path(gallery_path).exists():
        data = np.load(gallery_path, allow_pickle=True)
        gallery = {k: data[k] for k in data.files}

    gallery[label] = feat_vec

    Path(gallery_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(gallery_path, **gallery)

    # Save reference image
    ref_img_path = f"results/ref_{label}.jpg"
    cv2.imwrite(ref_img_path, frame)
    print(f"  OK - 「{label}」を登録しました (画像: {ref_img_path})")


def recognize_object(model, processor, source=6, gallery_path="results/gallery.npz", device="cuda"):
    """Live camera window with continuous DINOv3 recognition overlay. Press q to quit."""
    import cv2

    if not Path(gallery_path).exists():
        print("ERROR: ギャラリーが空です。先に --task register --label <名前> で物体を登録してください")
        return

    data = np.load(gallery_path, allow_pickle=True)
    gallery_labels = list(data.files)
    gallery_feats = torch.tensor(np.stack([data[k] for k in gallery_labels]))
    print(f"  登録済み物体: {gallery_labels}")
    print("  ライブ認識開始 (qで終了)")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera source={source}")
        return

    ranked = []
    frame_count = 0
    infer_every = 3  # Run DINOv3 every N frames to keep display smooth

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % infer_every == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            cls_feat, _ = extract_features(model, processor, pil_img, device)
            query = torch.nn.functional.normalize(cls_feat, dim=-1)
            sims = torch.mm(query, gallery_feats.T).squeeze(0)
            ranked = sorted(zip(gallery_labels, sims.tolist()), key=lambda x: -x[1])

        # Overlay results on frame
        display = frame.copy()
        y = 40
        for i, (label, score) in enumerate(ranked):
            color = (0, 255, 0) if i == 0 else (200, 200, 200)
            bar_len = int(score * 200)
            cv2.rectangle(display, (10, y - 18), (10 + bar_len, y), color, -1)
            text = f"{label}: {score:.2f}"
            cv2.putText(display, text, (220, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            y += 35

        cv2.putText(display, "DINOv3 Zero-Shot Recognition  [q: quit]",
                    (10, display.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 100), 1)
        cv2.imshow("DINOv3 Recognize", display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DINOv3 Demo")
    parser.add_argument("--image", default="bus.jpg", help="Input image path")
    parser.add_argument("--variant", default="vitb", choices=["vitb", "vitl"])
    parser.add_argument("--task", default="extract",
                        choices=["extract", "benchmark", "visualize", "camera", "register", "recognize"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--source", type=int, default=6, help="Camera device index (default: 6 for RealSense)")
    parser.add_argument("--label", default="object", help="Label name for --task register")
    args = parser.parse_args()

    model, processor = load_model(args.variant, args.device)

    if args.task == "extract":
        cls_feat, patch_feat = extract_features(model, processor, args.image, args.device)
        print(f"CLS feature shape: {cls_feat.shape}")
        print(f"Patch features shape: {patch_feat.shape}")
        print(f"CLS feature norm: {cls_feat.norm().item():.3f}")

    elif args.task == "benchmark":
        avg_ms, fps, vram_mb = benchmark_feature_extraction(model, processor, args.image, args.device)
        print(f"\nDINOv3 {args.variant} Benchmark:")
        print(f"  Avg: {avg_ms:.1f}ms | FPS: {fps:.1f} | VRAM: {vram_mb:.0f}MB")

    elif args.task == "visualize":
        visualize_patch_features(model, processor, args.image,
                                  save_path=f"results/dinov3_{args.variant}_features.png",
                                  device=args.device)

    elif args.task == "camera":
        camera_snapshot(model, processor, source=args.source,
                        save_dir="results", device=args.device, variant=args.variant)

    elif args.task == "register":
        register_object(model, processor, label=args.label,
                        source=args.source, device=args.device)

    elif args.task == "recognize":
        recognize_object(model, processor, source=args.source, device=args.device)
