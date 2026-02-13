from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI(title="Banana Death Predictor (TF)")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Model Variable
model = None

@app.on_event("startup")
def load_model():
    global model
    # Load the model once on startup
    try:
        model = tf.keras.models.load_model('banana_model.keras')
        print("TensorFlow Model Loaded Successfully")
    except Exception as e:
        print(f"Error loading model: {e}")

def preprocess_image(image_bytes):
    """
    Convert bytes to a format compatible with MobileNetV2 training
    (224x224, RGB, normalized 0-1)
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))
    image_array = np.array(image)
    
    # Normalize pixel values to [0, 1] as we did in training (rescale=1./255)
    image_array = image_array / 255.0
    
    # Add batch dimension: (224, 224, 3) -> (1, 224, 224, 3)
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

@app.get("/")
def home():
    return {"status": "Banana TensorFlow System Operational"}

@app.post("/predict")
async def predict_banana(file: UploadFile = File(...)):
    if not model:
        return {"error": "Model not loaded"}

    image_bytes = await file.read()
    processed_image = preprocess_image(image_bytes)
    
    # Predict
    prediction = model.predict(processed_image)
    
    # Extract float value
    days_left = float(prediction[0][0])
    
    # Logical capping
    days_left = max(0, days_left)

    return {
        "days_left": round(days_left, 1),
        "message": get_banana_message(days_left)
    }

def get_banana_message(days):
    if days > 7: return "Green and mean."
    if days > 3: return "Perfect eating window."
    if days > 1: return "EAT ME NOW."
    return "Banana Bread Time."