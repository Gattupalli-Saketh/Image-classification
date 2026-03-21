import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random


class RISE(nn.Module):
    """RISE (Randomized Input Sampling for Explanation) saliency.

    https://arxiv.org/abs/1806.07421
    """

    def __init__(self, model, input_size, device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        self.model = model.to(device)
        self.device = device
        self.input_size = input_size  # H, W
        self.C = 3  # assume 3 channels

    def generate_masks(self, N=3000, s=8, p1=0.1):
        """
        Step 1: Mask generation

        Generate N random binary masks at small scale (s × s),
        then upsample and smooth them.

        N    : number of masks
        s    : small mask size (e.g., 8×8)
        p1   : probability that a pixel in the small mask is 1 (kept)
        """
        H, W = self.input_size
        CH, CW = (H - 1) // s + 1, (W - 1) // s + 1  # ceil(H/s), ceil(W/s)

        # Upsampled mask size (smooth map)
        H_up, W_up = CH * s, CW * s

        masks = []
        for _ in range(N):
            # 1. Small binary mask
            mask = torch.rand(1, 1, s, s).to(self.device)
            mask = (mask < p1).float()  # 1 with probability p1

            # 2. Upscale via bilinear interpolation (smooth)
            #    to size (H_up, W_up) ≈ slightly larger than (H, W)
            mask = F.interpolate(mask, size=(H_up, W_up), mode="bilinear", align_corners=False)

            # 3. Random crop to match original input size (H, W)
            #    This is also part of mask generation.
            dy = random.randint(0, H_up - H)
            dx = random.randint(0, W_up - W)
            mask = mask[:, :, dy : dy + H, dx : dx + W]

            masks.append(mask)

        # return stacked masks: (N, 1, H, W)
        return torch.cat(masks, dim=0)

    def probing(self, image, masks, target_class=None):
        """
        Step 3: Probing the model

        Given:
          image  : (1, C, H, W)
          masks  : (N, 1, H, W)

        We create N occluded images:
          x_masked = image * mask   (foreground = kept, black elsewhere)

        Then run the model on all N images and collect:
          out[N, num_classes]

        For RISE, we keep the score for the target class only: (N,)
        """
        batch_size = 32  # avoid OOM

        N = masks.size(0)
        result = torch.zeros(N, device=self.device)

        for start_idx in range(0, N, batch_size):
            batch_masks = masks[start_idx : start_idx + batch_size]

            # Repeat image for each mask in the batch
            image_batch = image.expand(batch_masks.size(0), -1, -1, -1)  # (bs, C, H, W)

            # masked input: foreground pixels kept, others zeroed
            x_masked = image_batch * batch_masks  # (bs, C, H, W)

            with torch.no_grad():
                # model outputs probabilities (e.g., softmaxed)
                out = self.model(x_masked)  # (bs, num_classes)

            if target_class is None:
                # Use the predicted class
                target_class = out.argmax(dim=1)[0].item()

            # Keep score for the target class only
            result[start_idx : start_idx + batch_size] = out[:, target_class]

        return result  # (N,)

    def create_saliency_map(self, masks, scores):
        """
        Step 4: Saliency map creation

        RISE saliency map = weighted sum of masks:
          S = sum_i (score_i * mask_i) / sum_i score_i

        masks : (N, 1, H, W)
        scores: (N,)
        """
        # Normalize mask values to [0, 1] (they are already, but just in case)
        masks = masks.squeeze(1)  # (N, H, W)

        # Weighted sum of masks
        saliency = (scores.view(-1, 1, 1) * masks).sum(dim=0)

        # Normalize by total weight (optional but common)
        total_weight = scores.sum()
        if total_weight > 1e-8:
            saliency /= total_weight

        # Expand to (1, H, W) for consistency
        saliency = saliency.unsqueeze(0)

        return saliency

    def forward(self, image, N=3000, s=8, p1=0.1, target_class=None):
        """
        Complete RISE pipeline:

        1. Generate N binary masks at scale s and upsample/crop them.
        2. Probe the model on masked versions of the image.
        3. Build a saliency map as a weighted sum of masks.

        image       : (1, C, H, W), normalized, on device
        target_class: class index to explain (if None, use predicted class)

        Returns:
          saliency map (1, H, W)
        """
        # Step 1 & 2: Mask generation + Upscaling
        masks = self.generate_masks(N=N, s=s, p1=p1)

        # Step 3: Probing – collect model scores for masked images
        scores = self.probing(image, masks, target_class=target_class)

        # Step 4: Saliency map from masks and scores
        saliency = self.create_saliency_map(masks, scores)

        return saliency


