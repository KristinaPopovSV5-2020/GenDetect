import io
from unittest import case
import cv2
import torch
from PIL import Image
import uvicorn
import numpy as np
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse

from resnet50.gradCAM import GradCAM, overlay_heatmap
from resnet50.model import ResNetClassifier
from fastapi import FastAPI, UploadFile, File, Form

from resnet50.train_config import TrainResNetConfig
from transforms import val_transforms

from vit.vit import Config, build_model
from vit.xai import ViTAttentionRollout, overlay_heatmap, ViTGradCAM

HOST = "0.0.0.0"
PORT = 8000
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["predict", "gradcam"],
    allow_headers=["*"],
)
device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')


# ResNet
best_config_resnet = TrainResNetConfig(unfreeze_layers=("layer3", "layer4"),dropout=0.6, use_scheduler=False, model_name='resnet50-best-config.pth')
resnet_model = ResNetClassifier(best_config_resnet)
checkpoint = torch.load(
    f"resnet50/{best_config_resnet.model_name}",
    map_location=device,
    weights_only=False
)

resnet_model.load_state_dict(checkpoint["model_state"])
threshold_resnet = float(checkpoint["threshold"])
resnet_model.to(device)
resnet_model.eval()

target_resnet_layer = getattr(resnet_model.backbone, "layer4")[-1]
resnet_gradcam_model = GradCAM(resnet_model, target_resnet_layer)

# ViT
vit_config = Config()
vit_model = build_model(vit_config)
vit_checkpoint = torch.load("../outputs/best_vit_model.pt", map_location=vit_config.device)
vit_model.load_state_dict(vit_checkpoint["model_state_dict"])
vit_model.to(vit_config.device)
vit_model.eval()

vit_attention_model = ViTAttentionRollout(vit_model)
vit_gradcam_model = ViTGradCAM(vit_model, target_block_idx=-2)

models = {
    "resnet": resnet_model,
    "vit": vit_model,
}

xai_models = {
    "vit_attention": vit_attention_model,
    "vit_gradcam": vit_gradcam_model,
    "resnet_gradcam": resnet_gradcam_model
}

thresholds = {
    "resnet": threshold_resnet,
    "vit": 0.585,
}

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model_name: str = Form(...)
):
    if model_name not in models:
        return {"error": "Invalid model"}

    model = models[model_name]
    threshold = thresholds[model_name]

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    input_tensor = val_transforms(image).unsqueeze(0).to(device)

    with torch.no_grad():
        if model_name == "resnet":
            # internal convention: 0 = fake, 1 = real
            output = model(input_tensor).view(-1)
            prob_real = torch.sigmoid(output).item()
            prob_fake = 1.0 - prob_real
            pred = int(prob_fake >= (1.0 - threshold))

        elif model_name == "vit":
            # internal convention: 1 = fake
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1)
            prob_fake = probs[0, 1].item()
            pred = int(prob_fake >= threshold)

        else:
            return {"error": "Unsupported model"}

    return {
        "model": model_name,
        "prediction": pred,            
        "probability_fake": prob_fake, 
        "threshold": threshold
    }

@app.post("/xai")
async def xai(file: UploadFile = File(...), model_name: str = Form(...)):

    if model_name not in xai_models:
        return {"error": "Invalid XAI model"}

    model = xai_models[model_name]

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    image_np = np.array(image)

    input_tensor = val_transforms(image).unsqueeze(0).to(device)
    input_tensor.requires_grad = True

    heatmap = model.generate(input_tensor)

    output_img = overlay_heatmap(image_np, heatmap)

    _, buffer = cv2.imencode(".jpg", output_img)

    return StreamingResponse(
        io.BytesIO(buffer.tobytes()),
        media_type="image/jpeg"
    )

if __name__ == "__main__":
    uvicorn.run("server:app", host=HOST, port=PORT, reload=True)
