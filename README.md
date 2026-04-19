# GenDetect: AI vs Real Image Detection with XAI 🕵🏽‍♀️

GenDetect is a deep learning system for detecting whether an image is **AI-generated or real**, using and comparing two popular architectures:

- **ResNet50 (CNN)**
- **Vision Transformer (ViT)**

The project also focuses on **Explainable AI (XAI)**:
- Grad-CAM (ResNet & ViT)
- Attention Rollout (ViT)

## Setup Instructions

### Model setup

### 1. Download checkpoints
You can download the fine-tuned model checkpoints [here](https://drive.google.com/drive/folders/1n6aQ23kmzSl8nTu1Dt7_yrDZ5eZuK3kp).

### 2. Add to .env
Create *.env* file in `backend` folder.
Add the path to the folder where you downloaded the model checkpoints to the .env file:

```bash
MODEL_FOLDER=example/
```

### Backend Setup (FastAPI)

#### 1. Create virtual environment

```bash
cd backend
python3 -m venv venv
source venv/bin/activate     # Mac/Linux
venv\Scripts\activate        # Windows
```

#### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 2.5 Download Dataset [OPTIONAL]

To run model training, add route to dataset folder to the *.env* file:

```bash
INPUT_DATA_FOLDER=/example      #path to your downloaded dataset
OUTPUT_DATASET_FOLDER=/example  #path where the script will store the train, test, and validation splits
```

The scripts expect a dataset folder with the following structure:
```bash
├── INPUT_DATA_FOLDER/
│ ├── real/
│ └── fake/
```

#### 3. Run server

```bash
uvicorn server:app --reload
```

### Frontend Setup (React + Vite)

#### 1. Install dependencies

```bash
cd frontend
npm install
```

#### 2. Install dependencies

```bash
npm run dev
```

## Usage

1. Open frontend UI  
2. Upload an image  
3. System outputs **"Detection Results"** for both CNN & ViT:  
   - Prediction (Fake / Real)  
   - Probability (confidence)
   - Decision threshold
4. **XAI** components shown:  
   - Grad-CAM (ResNet)  
   - Grad-CAM (ViT)  
   - Attention Rollout (ViT)

## Authors 👩‍💻
- [Kristina Popov](https://github.com/KristinaPopovSV5-2020)
- [Tina Mihajlovic](https://github.com/tince250)

MSc Course: **Neural Networks**