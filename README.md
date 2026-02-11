# SmartVision Engine: Adaptive Detection & Automated Retraining Pipeline

SmartVision Engine is an advanced computer vision framework designed for iterative object recognition. It leverages a Human-in-the-Loop (HITL) approach, allowing users to validate and correct AI detections in real-time to trigger an automated model evolution.

## Technologies and Libraries Used

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-blue?style=for-the-badge)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Jinja2](https://img.shields.io/badge/Jinja2-B41717?style=for-the-badge&logo=jinja&logoColor=white)

* **Ultralytics YOLOv8:** Base model for object detection.
* **MLflow:** Model lifecycle management, experiment tracking, and versioning.
* **FastAPI & Uvicorn:** Asynchronous server to expose the model via REST API.
* **OpenCV & Matplotlib:** Image processing and visualization.
* **Roboflow:** Dataset management and versioning.

## Method Overview

The SmartVision Engine implements a Human-in-the-Loop architectural pattern that synchronizes real-time detection with an automated MLOps pipeline. Upon image ingestion, the system executes inference via YOLOv8, applying a custom mapping layer to normalize labels into business-specific categories. When a user corrects a detection, the frontend converts pixel coordinates into YOLO-normalized format, generating high-quality ground truth data. This feedback triggers a background transfer learning process, where the model fine-tunes its weights using the new samples while MLflow tracks performance metrics. Finally, the system performs a hot-swap deployment, replacing the active model with the optimized best.pt to ensure continuous precision improvement without service interruption.

## Target Classes (Multilabel)

The model predicts **independent probabilities** for each class:

| Class | Description |
|-------|-------------|
| `sofa` | Large seating furniture including couches and armchairs |
| `rug` | Area rugs, carpets, and textile floor coverings |
| `pillows`| Support accessories, cushions, and comfort elements |

## Technical Details

* **Class Independence:** Being Multilabel, the system can detect a rug under a sofa and pillows on top of it without the probabilities of one affecting the other.
* **Success Metric:** The model uses a Binary Cross-Entropy loss function, which allows each label to be treated as an independent “yes/no” problem.

## Project Structure

```text
mi_proyecto_vision/
|
├── app/                        # Main Application Core
│   ├── main.py                 # FastAPI Server & Model Orchestration [cite: 9, 15]
│   ├── retrain_service.py      # Background Retraining Logic [cite: 13, 15]
│   ├── templates/              # UI Components (index.html) [cite: 15]
│   └── __init__.py             # Package initialization
│
├── data/                       # Dataset Management [cite: 13]
│   ├── feedback_dataset/       # Stored user corrections (labels/images) [cite: 12]
│   ├── temp_uploads/           # Buffer for incoming inference images
│   └── retrain_config.yaml     # Dynamic YOLO configuration file [cite: 15]
│
├── models/                     # Model Weights & Registry [cite: 15]
│   ├── best.pt                 # Current Production Model [cite: 10]
│   ├── best_v1769803043.pt     # Timestamped Version (Hot-Swap ready) [cite: 46]
│   └── test/                   # Validation weights
│
├── Notebooks/                  # Experimental Phase & EDA
│   ├── Dataset_Base01.ipynb    # Initial data processing [cite: 15]
│   ├── Entrenamiento02.ipynb   # Training experiments [cite: 15]
│   └── Prediction03.ipynb      # Inference testing and metrics [cite: 15]
│
├── models_history/             # MLflow local tracking artifacts [cite: 15, 16]
├── short_paper/                # Documentation & Research Paper [cite: 1]
└── README.md                   # Project Overview & Documentation
```
## Installation and Execution

### 1. Clone the repository and install dependencies
```bash
git clone <URL_DEL_REPO>
cd /folderproyect

python -m venv venv
.\venv\Scripts\activate
pip install -r
ultralytics
roboflow
opencv-python
matplotlib
seaborn
pandas
mlflow
fastapi
uvicorn
python-multipart
sqlalchemy
jinja2
ipykernel
pyyaml
pydantic
requests
```
### 2. Setup de GPU

You can use the following script to let the model use your GPU instead of CPU.
```bash
setup_gpu.bat
```
### 3. Run the Application

You can use the automatic script on Windows:
```bash

uvicorn app.main:app --reload
```

### 4. Access the Web
Open the browser at: `localhost:8000`

---
### 5. GitHub
https://github.com/JorgePizarroL/mi_proyecto_vision.git

## Authors

* **Jorge Pizarro** - [Github](https://github.com/JorgePizarroL)  
  Contact: +593 98 594 8662 - jpizarrol1@est.ups.edu.ec
* **Jonathan Morocho** - [Github](https://github.com/jonathanMorochoYu)  
  Contact: +593 98 741 3828 - jmorochoy2@est.ups.edu.ec 
