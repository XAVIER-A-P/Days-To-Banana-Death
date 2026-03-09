# 🍌 Days to Banana Death: End-to-End ML Pipeline

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)
![Status](https://img.shields.io/badge/Status-Deployed-success.svg)

**Days to Banana Death** is a production-grade, full-stack Machine Learning web application that uses Computer Vision to predict the remaining shelf life (in days) of a banana. 

Unlike standard classification tasks (e.g., "ripe" vs. "unripe"), this project tackles a **Continuous Regression Problem** requiring a custom data collection pipeline, efficient edge-friendly model selection, and a decoupled API architecture.

<br>

<p align="center">
  <img src="assets/app_home.png" alt="Home Screen" width="250"/>
  &nbsp;&nbsp;&nbsp;
  <img src="assets/app_scanning.png" alt="Scanning Animation" width="250"/>
  &nbsp;&nbsp;&nbsp;
  <img src="assets/app_result.png" alt="Prediction Result" width="250"/>
</p>

---

## 🏗️ System Architecture & Engineering Focus

This project was built to demonstrate an understanding of the **entire Machine Learning Lifecycle**, moving beyond notebooks and into production systems.

1. **Data Engineering (Custom Dataset):** Engineered a custom dataset by photographing bananas at consistent intervals, normalizing for lighting, and labeling temporal regression targets (days until defined "death").
2. **Model Selection for Production:** Utilized Transfer Learning with **MobileNetV2**. Chosen specifically over heavier architectures (like ResNet50) to minimize the Docker image footprint and reduce inference latency/costs on CPU-only cloud instances.
3. **Decoupled Architecture:** Built a RESTful backend using **FastAPI** that serves inference via an HTTP endpoint. The frontend is a lightweight, framework-agnostic HTML/JS client that handles real-time camera streaming and base64/blob image processing.
4. **Containerization:** The application is fully containerized using **Docker**, isolating dependencies (like `tensorflow-cpu`) and ensuring environment parity between local development and cloud deployment.

---

## 🛠️ Tech Stack

* **Machine Learning:** TensorFlow, Keras, OpenCV / Pillow, Pandas, NumPy.
* **Backend:** FastAPI, Uvicorn, Python.
* **Frontend:** Vanilla JavaScript, HTML5, CSS3, WebRTC (MediaDevices API).
* **Deployment & DevOps:** Docker, Render (Cloud PaaS), Git.

---

## 🚀 Live Demo

* **Web App:** https://days-to-banana-death-4.onrender.com
* *(Note: Allow 30-50 seconds for the server to spin up if it has been idle, as it is hosted on a free cloud tier.)*

---

## 📂 Project Structure

```
├── data/                         # Scripts and guidelines for data collection
├── training/                     # Jupyter notebooks and model training scripts
│   ├── train.py                  # Model architecture and training pipeline
│   └── data_augmentation.py      # Logic for robust training
├── app/                          # Production Deployment Zone
│   ├── static/                   # Frontend assets
│   │   └── index.html            # WebRTC camera UI
│   ├── main.py                   # FastAPI backend
│   ├── banana_model.keras        # Compiled and optimized model weights
│   ├── requirements.txt          # Explicit pip dependencies
│   └── Dockerfile                # Production container instructions
└── README.md
```

---

## 🔮 Future Improvements
* **Automated CI/CD:** Implement GitHub Actions to automatically lint code, run unit tests on the API, and push the latest Docker image to Render upon merges to `main`.
* **Data Drift Monitoring:** Implement shadow logging to save user-uploaded bananas (with consent) to an S3 bucket to analyze real-world data distribution vs. training data.
* **Model Quantization:** Convert the `.keras` model to TensorFlow Lite to further reduce inference time and memory usage.
