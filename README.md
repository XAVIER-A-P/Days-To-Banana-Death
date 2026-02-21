# 🍌 Days to Banana Death: End-to-End ML Pipeline

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)
![Status](https://img.shields.io/badge/Status-Deployed-success.svg)

**Days to Banana Death** is a production-grade, full-stack Machine Learning web application that uses Computer Vision to predict the remaining shelf life (in days) of a banana. 

Unlike standard classification tasks (e.g., "ripe" vs. "unripe"), this project tackles a **Continuous Regression Problem** requiring a custom data collection pipeline, efficient edge-friendly model selection, and a decoupled API architecture.

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

* **Web App:** [Insert your Render URL here, e.g., https://days-to-banana-death-1.onrender.com]
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

## 💻 Running the Project Locally

### Prerequisites
* Python 3.9+
* Docker (optional, but recommended)

### Option 1: Using Docker (Recommended)
1. Clone the repository:
   \`\`\`bash
   git clone https://github.com/YOUR_USERNAME/days-to-banana-death.git
   cd days-to-banana-death/app
   \`\`\`
2. Build the Docker image:
   \`\`\`bash
   docker build -t banana-death-app .
   \`\`\`
3. Run the container:
   \`\`\`bash
   docker run -p 8000:8000 banana-death-app
   \`\`\`
4. Open your browser and navigate to \`http://localhost:8000\`.

### Option 2: Using Python Virtual Environment
1. Clone and enter the app directory:
   \`\`\`bash
   git clone https://github.com/YOUR_USERNAME/days-to-banana-death.git
   cd days-to-banana-death/app
   \`\`\`
2. Create and activate a virtual environment:
   \`\`\`bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   \`\`\`
3. Install dependencies:
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`
4. Run the Uvicorn server:
   \`\`\`bash
   uvicorn main:app --reload
   \`\`\`

---

## 🔮 Future Improvements
* **Automated CI/CD:** Implement GitHub Actions to automatically lint code, run unit tests on the API, and push the latest Docker image to Render upon merges to `main`.
* **Data Drift Monitoring:** Implement shadow logging to save user-uploaded bananas (with consent) to an S3 bucket to analyze real-world data distribution vs. training data.
* **Model Quantization:** Convert the `.keras` model to TensorFlow Lite to further reduce inference time and memory usage.
