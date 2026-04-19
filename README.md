# GenDetect: AI vs Real Image Detection with XAI 🕵🏽‍♀️

GenDetect is a deep learning system for detecting whether an image is **AI-generated or real**, using and comparing two popular architectures:

- **ResNet50 (CNN)**
- **Vision Transformer (ViT)**

The project also focuses on **Explainable AI (XAI)**:
- Grad-CAM (ResNet & ViT)
- Attention Rollout (ViT)

## Setup Instructions

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