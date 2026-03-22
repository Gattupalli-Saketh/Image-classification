import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


# ── Dummy model (used when no trained model exists) ───────
class DummyClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.AdaptiveAvgPool2d(1)
        )
        self.fc = torch.nn.Linear(64, 10)  # 10 dummy classes

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ── SIFT keypoints detection ──────────────────────────────
def detect_sift_keypoints(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create()
    kp = sift.detect(gray, None)
    print(f"Detected {len(kp)} SIFT keypoints")
    return kp


# ── Generate RISE-style masks guided by SIFT ──────────────
def generate_sift_masks(img_shape, kp, N=800, grid_size=56):
    H, W = img_shape[:2]
    masks = np.zeros((N, H, W), dtype=np.float32)

    for i in range(N):
        mask = np.zeros((H, W), dtype=np.float32)
        if len(kp) == 0:
            continue

        num_kp_use = np.random.randint(1, min(12, len(kp) + 1))
        selected_idx = np.random.choice(len(kp), num_kp_use, replace=False)
        selected_kp = [kp[j] for j in selected_idx]

        for k in selected_kp:
            x, y = int(k.pt[0]), int(k.pt[1])
            radius = max(int(k.size * 0.7), 5)
            yy, xx = np.ogrid[:H, :W]
            sigma = radius / 2.2
            blob = np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * sigma**2))
            mask = np.maximum(mask, blob)

        # RISE-style down→up sampling
        small = cv2.resize(mask, (grid_size, grid_size), interpolation=cv2.INTER_NEAREST)
        up = cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)
        masks[i] = up

    # Normalize masks (sum ≈ 1 per mask)
    sums = masks.sum(axis=(1, 2), keepdims=True)
    masks = np.divide(masks, np.maximum(sums, 1e-6))
    return masks


# ── Load and preprocess image ─────────────────────────────
def load_image(image_path, size=224):
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"cv2.imread failed on {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensor = transform(img_rgb).unsqueeze(0)
    return tensor, img_rgb


# ── Compute RISE saliency (FIXED) ─────────────────────────
def compute_rise_saliency(model, img_tensor, masks, target_class=None):
    """
    masks is expected to already be resized to model input size (N, 224, 224)
    """
    model.eval()
    device = next(model.parameters()).device
    img_tensor = img_tensor.to(device)

    scores = []
    batch_size = 80

    with torch.no_grad():
        for i in range(0, len(masks), batch_size):
            batch_np = masks[i:i + batch_size]
            batch_t = torch.from_numpy(batch_np).float().unsqueeze(1).to(device)  # (bs,1,H,W)

            masked = img_tensor.repeat(batch_t.shape[0], 1, 1, 1) * batch_t

            preds = model(masked)
            probs = F.softmax(preds, dim=1)

            if target_class is None:
                batch_scores = probs.max(dim=1)[0].cpu().numpy()
            else:
                batch_scores = probs[:, target_class].cpu().numpy()

            scores.append(batch_scores)

    scores = np.concatenate(scores)

    # === Correct RISE formula ===
    # saliency = mean over i ( mask_i * score_i )
    saliency = np.zeros_like(masks[0])  # (224, 224)
    for m, s in zip(masks, scores):
        saliency += m * s

    saliency /= len(scores)   # or /= scores.sum() if you want probability-weighted

    return saliency, scores


# ── Main execution ────────────────────────────────────────
def main():
    image_path = "heroine.jpg"          # ← make sure this file exists!

    print("1. Loading image...")
    try:
        img_tensor, img_rgb_orig = load_image(image_path)
    except Exception as e:
        print(f"Failed: {e}")
        return

    # Visualization-sized version (224×224)
    img_rgb = cv2.resize(img_rgb_orig, (224, 224), interpolation=cv2.INTER_LINEAR)

    print("2. Detecting SIFT keypoints...")
    kp = detect_sift_keypoints(img_rgb_orig)

    print("3. Generating SIFT-guided masks...")
    masks_orig = generate_sift_masks(img_rgb_orig.shape, kp, N=800)

    print("   → Resizing masks to 224×224")
    masks = np.array([
        cv2.resize(m, (224, 224), interpolation=cv2.INTER_LINEAR)
        for m in masks_orig
    ])
    sums = masks.sum(axis=(1,2), keepdims=True)
    masks /= np.maximum(sums, 1e-8)

    print(f"   Final masks shape: {masks.shape}")

    print("4. Creating model...")
    model = DummyClassifier()

    model_path = "your_model.pth"
    if os.path.isfile(model_path):
        try:
            state = torch.load(model_path, map_location='cpu', weights_only=True)
            model.load_state_dict(state, strict=False)
            print("   → Loaded trained model")
        except Exception as e:
            print(f"   Model loading failed: {e}")
            print("   → Using random weights")
    else:
        print("   No model file found → using random (untrained) weights")

    print("5. Computing saliency map...")
    saliency, scores = compute_rise_saliency(model, img_tensor, masks)

    print("6. Creating visualization...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.ravel()

    axes[0].imshow(img_rgb)
    axes[0].set_title("Original (resized to 224×224)")
    axes[0].axis('off')

    if len(kp) > 0:
        vis_orig = cv2.drawKeypoints(img_rgb_orig, kp, None,
                                     flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        vis = cv2.resize(vis_orig, (224, 224), interpolation=cv2.INTER_AREA)
        axes[1].imshow(vis)
    else:
        axes[1].imshow(img_rgb)
    axes[1].set_title(f"SIFT keypoints ({len(kp)})")
    axes[1].axis('off')

    axes[2].imshow(masks[0], cmap='gray')
    axes[2].set_title("Sample mask")
    axes[2].axis('off')

    axes[3].imshow(saliency, cmap='jet')
    axes[3].set_title("Saliency map")
    axes[3].axis('off')

    overlay = 0.35 * (img_rgb.astype(float) / 255.0) + 0.65 * np.stack([saliency] * 3, axis=-1)
    overlay = np.clip(overlay, 0, 1)
    axes[4].imshow(overlay)
    axes[4].set_title("Overlay")
    axes[4].axis('off')

    axes[5].hist(scores, bins=50, color='coral', alpha=0.7)
    axes[5].set_title(f"Scores (mean = {scores.mean():.4f})")
    axes[5].set_xlabel("Score")

    plt.tight_layout()
    out_file = "sift_rise_result.jpg"
    plt.savefig(out_file, dpi=180, bbox_inches='tight')
    plt.close(fig)

    print(f"\nDone! Result saved → {out_file}")
    print(f"Peak saliency value: {saliency.max():.4f}")


if __name__ == "__main__":
    main()