import math
import torch
import torch.nn.functional as F
import numpy as np
import cv2


def get_backbone(model):
    return model.backbone if hasattr(model, "backbone") else model


def overlay_heatmap(img, heatmap):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(
        np.array(255 * heatmap, dtype=np.uint8),
        cv2.COLORMAP_JET
    )
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    return superimposed_img


class ViTAttentionRollout:
    def __init__(self, model):
        self.model = model
        self.backbone = get_backbone(model)
        self._patch_attention_modules()

    def _patch_attention_modules(self):
        for block in self.backbone.blocks:
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

    def _get_attention_maps(self):
        maps = []
        for block in self.backbone.blocks:
            if not hasattr(block.attn, "attn_map"):
                raise ValueError("Missing attention maps. Run a forward pass first.")
            maps.append(block.attn.attn_map)
        return maps

    def _compute_rollout(self, attn_maps):
        device = attn_maps[0].device
        tokens = attn_maps[0].size(-1)
        batch_size = attn_maps[0].size(0)

        result = torch.eye(tokens, device=device).unsqueeze(0).repeat(batch_size, 1, 1)

        for attn in attn_maps:
            attn = attn.mean(dim=1)  # average heads
            eye = torch.eye(tokens, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
            attn = attn + eye
            attn = attn / attn.sum(dim=-1, keepdim=True)
            result = attn @ result

        return result

    def generate(self, input_tensor):
        self.model.eval()

        with torch.no_grad():
            _ = self.model(input_tensor)

        attn_maps = self._get_attention_maps()
        rollout = self._compute_rollout(attn_maps)  # [B, tokens, tokens]

        mask = rollout[0, 0, 1:]  # CLS -> patches
        num_patches = mask.shape[0]

        grid_size = int(math.sqrt(num_patches))
        if grid_size * grid_size != num_patches:
            raise ValueError(f"Patch count {num_patches} is not a square number")

        heatmap = mask.reshape(grid_size, grid_size).detach().cpu().numpy()
        heatmap = np.maximum(heatmap, 0)

        if heatmap.max() != 0:
            heatmap /= heatmap.max()

        return heatmap


class ViTGradCAM:
    def __init__(self, model, target_block_idx=-2):
        self.model = model
        self.backbone = get_backbone(model)
        self.target_layer = self.backbone.blocks[target_block_idx].norm1

        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output  # [B, tokens, C]

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]  # [B, tokens, C]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def _reshape_transform(self, tensor):
        # tensor: [B, tokens, C]
        tensor = tensor[:, 1:, :]  # remove CLS token

        B, num_patches, C = tensor.shape
        grid_size = int(math.sqrt(num_patches))
        if grid_size * grid_size != num_patches:
            raise ValueError(f"Patch count {num_patches} is not a square number")

        tensor = tensor.reshape(B, grid_size, grid_size, C)   # [B, H, W, C]
        tensor = tensor.permute(0, 3, 1, 2)                   # [B, C, H, W]
        return tensor

    def generate(self, input_tensor):
        self.model.eval()
        self.model.zero_grad()

        output = self.model(input_tensor)

        # multi-class and binary-safe for your 2-logit setup
        class_idx = output.argmax(dim=1)
        score = output[torch.arange(output.size(0)), class_idx]
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise ValueError("Grad-CAM failed: missing gradients or activations")

        activations = self._reshape_transform(self.activations)  # [B, C, H, W]
        gradients = self._reshape_transform(self.gradients)      # [B, C, H, W]

        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)   # [B, C, 1, 1]
        cam = torch.sum(weights * activations, dim=1)[0]            # [H, W]

        heatmap = cam.detach().cpu().numpy()
        heatmap = np.maximum(heatmap, 0)

        if heatmap.max() != 0:
            heatmap /= heatmap.max()

        return heatmap