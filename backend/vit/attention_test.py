import math
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

from vit.vit import build_model, Config


def get_backbone(model):
    return model.backbone if hasattr(model, "backbone") else model


def load_model(checkpoint_path: str):
    cfg = Config()
    model = build_model(cfg)

    checkpoint = torch.load(checkpoint_path, map_location=cfg.device)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(cfg.device)
    model.eval()
    return model, cfg, checkpoint


def preprocess_image(image_path: str, image_size: int):
    image = Image.open(image_path).convert("RGB")
    original_img = np.array(image)

    image = image.resize((image_size, image_size))
    img_np = np.array(image).astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_np = (img_np - mean) / std

    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float()
    return img_tensor, original_img

def enable_attention_extraction(model):

    backbone = get_backbone(model)

    for block in backbone.blocks:
        attn_module = block.attn

        def new_forward(x, attn_mask=None, is_causal=False, attn_module=attn_module):
            B, N, C = x.shape

            qkv = attn_module.qkv(x).reshape(
                B, N, 3, attn_module.num_heads, C // attn_module.num_heads
            )
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)

            if hasattr(attn_module, "q_norm") and attn_module.q_norm is not None:
                q = attn_module.q_norm(q)
            if hasattr(attn_module, "k_norm") and attn_module.k_norm is not None:
                k = attn_module.k_norm(k)

            q = q * attn_module.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = attn_module.attn_drop(attn)

            attn_module.attn_map = attn.detach()

            x = attn @ v
            x = x.transpose(1, 2).reshape(B, N, C)
            x = attn_module.proj(x)
            x = attn_module.proj_drop(x)
            return x

        attn_module.forward = new_forward


def get_attention_maps(model):
    backbone = get_backbone(model)
    maps = []

    for block in backbone.blocks:
        if not hasattr(block.attn, "attn_map"):
            raise RuntimeError(
                "Attention map not found."
            )
        maps.append(block.attn.attn_map)

    return maps


def compute_rollout(attn_maps):
    device = attn_maps[0].device
    tokens = attn_maps[0].size(-1)
    batch_size = attn_maps[0].size(0)

    result = torch.eye(tokens, device=device).unsqueeze(0).repeat(batch_size, 1, 1)

    for attn in attn_maps:
        attn = attn.mean(dim=1)  
        eye = torch.eye(tokens, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        attn = attn + eye
        attn = attn / attn.sum(dim=-1, keepdim=True)
        result = attn @ result

    return result


class ViTGradCAM:
    def __init__(self, model, target_block_idx=-1):
        self.model = model
        self.backbone = get_backbone(model)
        self.target_block = self.backbone.blocks[target_block_idx]

        self.activations = None
        self.gradients = None

        self.fwd_handle = self.target_block.register_forward_hook(self._forward_hook)
        self.bwd_handle = self.target_block.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output  

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]  

    def remove(self):
        self.fwd_handle.remove()
        self.bwd_handle.remove()

    def generate(self, class_idx: int):
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Run forward + backward before calling generate().")

        acts = self.activations[0]   
        grads = self.gradients[0]    

        acts = acts[1:, :]           
        grads = grads[1:, :]         

        print("Grad mean:", grads.abs().mean().item())
        print("Act mean:", acts.abs().mean().item())

        weights = grads.mean(dim=0) 

        cam = acts @ weights       
        cam = torch.relu(cam)

        num_patches = cam.shape[0]
        grid_size = int(math.sqrt(num_patches))
        if grid_size * grid_size != num_patches:
            raise ValueError(f"Patch count {num_patches} is not a square number")

        cam = cam.reshape(grid_size, grid_size)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.detach().cpu().numpy()

def resize_mask_to_image(mask, original_img):
    mask_tensor = torch.tensor(mask).unsqueeze(0).unsqueeze(0).float()
    resized = F.interpolate(
        mask_tensor,
        size=(original_img.shape[0], original_img.shape[1]),
        mode="bilinear",
        align_corners=False
    )[0, 0].numpy()
    return resized


def overlay_attention(original_img, mask):
    mask = cv2.resize(mask, (original_img.shape[1], original_img.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)
    return overlay

def explain_image(model, cfg, image_path, threshold=0.5, gradcam_block_idx=-1):
    device = cfg.device

    img_tensor, original_img = preprocess_image(image_path, cfg.image_size)
    img_tensor = img_tensor.to(device)

    enable_attention_extraction(model)

    gradcam = ViTGradCAM(model, target_block_idx=gradcam_block_idx)

    model.zero_grad()
    outputs = model(img_tensor)
    print(outputs)
    if outputs.shape[-1] == 1:
        prob_fake = float(outputs.squeeze().item())
        prob_real = 1.0 - prob_fake
        fake_score = outputs.squeeze()
    else:
        probs = F.softmax(outputs, dim=1)[0]
        prob_real = float(probs[0].item())
        prob_fake = float(probs[1].item())
        fake_score = outputs[0, 1] 

    pred = 1 if prob_fake >= threshold else 0
    label = "Fake" if pred == 1 else "Real"

    print(f"\nPrediction: {label}")
    print(f"Probability (real): {prob_real:.4f}")
    print(f"Probability (fake): {prob_fake:.4f}")
    print(f"Threshold: {threshold:.3f}")

    fake_score.backward(retain_graph=True)

    attn_maps = get_attention_maps(model)
    rollout = compute_rollout(attn_maps)  
    rollout_mask = rollout[0, 0, 1:].detach().cpu().numpy()

    size = int(math.sqrt(rollout_mask.shape[0]))
    if size * size != rollout_mask.shape[0]:
        raise ValueError(f"Patch count {rollout_mask.shape[0]} is not a square number")

    rollout_mask = rollout_mask.reshape(size, size)
    rollout_mask = (rollout_mask - rollout_mask.min()) / (rollout_mask.max() - rollout_mask.min() + 1e-8)

    gradcam_mask = gradcam.generate(class_idx=1)

    rollout_overlay = overlay_attention(original_img, rollout_mask)
    gradcam_overlay = overlay_attention(original_img, gradcam_mask)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(original_img)
    axes[0, 0].set_title("Original")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(rollout_mask, cmap="jet")
    axes[0, 1].set_title("Attention Rollout")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(cv2.cvtColor(rollout_overlay, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title(f"Rollout Overlay | {label} | P(fake)={prob_fake:.3f}")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(original_img)
    axes[1, 0].set_title("Original")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(gradcam_mask, cmap="jet")
    axes[1, 1].set_title("Transformer Grad-CAM")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(cv2.cvtColor(gradcam_overlay, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title("Grad-CAM Overlay")
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.show()

    gradcam.remove()

    return {
        "predicted_label": label,
        "prob_real": prob_real,
        "prob_fake": prob_fake,
        "rollout_mask": rollout_mask,
        "gradcam_mask": gradcam_mask,
    }


if __name__ == "__main__":
    model_path = "outputs/best_vit_model.pt"
    test_dir: str = "/Users/tinamihajlovic/.cache/kagglehub/datasets/tristanzhang32/ai-generated-images-vs-real-images/versions/2/test"          

    # 3602, 3603
    image_path = f"{test_dir}/fake/4905.jpg"
    model, cfg, checkpoint = load_model(model_path)
    threshold = checkpoint.get("best_threshold", 0.5)

    explain_image(
        model=model,
        cfg=cfg,
        image_path=image_path,
        threshold=threshold,
        gradcam_block_idx=-2,   # last block is idx 11
    )