import torch
import numpy as np

class GradCAM:

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = []
        self.activations = []

        self.register_hooks()


    def register_hooks(self):

        def forward_hook(module, input, output):
            self.activations.append(output)

        def backward_hook(module, grad_input, grad_output):
            self.gradients.append(grad_output[0])

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)


    def generate(self, input_tensor):
        self.model.zero_grad() # Zero the gradients
        output = self.model(input_tensor)
        pred_class = output.argmax(dim=1).item()

        # Backward pass to compute gradients
        output[:, pred_class].backward()

        # Compute the weights
        weights = torch.mean(self.gradients[0], dim=[2, 3], keepdim=True)

        cam = torch.sum(weights * self.activations[0], dim=1)
        heatmap = cam.squeeze()
        heatmap = np.maximum(heatmap.cpu().detach().numpy(), 0)
        heatmap /= np.max(heatmap)
        return heatmap