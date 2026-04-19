import io
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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
gradcam_model = GradCAM(resnet_model, target_resnet_layer)

print(threshold_resnet)
models = {
    "resnet": resnet_model,
}

thresholds = {
    "resnet": threshold_resnet,
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
        output = model(input_tensor)
        prob = torch.sigmoid(output).item()
        pred = int(prob >= threshold)

    return {
        "model": model_name,
        "prediction": pred,
        "probability": prob
    }

@app.post("/gradcam")
async def gradcam(file: UploadFile = File(...)):

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    image_np = np.array(image)

    input_tensor = val_transforms(image).unsqueeze(0).to(device)
    input_tensor.requires_grad = True

    heatmap = gradcam_model.generate(input_tensor)

    output_img = overlay_heatmap(image_np, heatmap)

    _, buffer = cv2.imencode(".jpg", output_img)

    return StreamingResponse(
        io.BytesIO(buffer.tobytes()),
        media_type="image/jpeg"
    )

if __name__ == "__main__":
    uvicorn.run("server:app", host=HOST, port=PORT, reload=True)
