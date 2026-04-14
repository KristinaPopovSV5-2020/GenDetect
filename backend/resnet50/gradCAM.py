import torch
import numpy as np
import cv2

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor):
        self.model.eval()
        self.model.zero_grad()

        output = self.model(input_tensor)

        # target class (binary or multi-class safe)
        class_idx = output.argmax(dim=1)

        score = output[:, class_idx]
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise ValueError("Grad-CAM failed: missing gradients or activations")

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1)[0]

        heatmap = cam.detach().cpu().numpy()
        heatmap = np.maximum(heatmap, 0)

        if heatmap.max() != 0:
            heatmap /= heatmap.max()

        return heatmap


def overlay_heatmap(img, heatmap):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    heatmap = cv2.applyColorMap(
        np.array(255 * heatmap, dtype=np.uint8),
        cv2.COLORMAP_JET
    )

    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    return superimposed_img