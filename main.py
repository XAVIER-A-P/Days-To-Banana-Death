from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI()

# 1. MOUNT THE STATIC FOLDER
# This allows the app to see the folder where we put index.html
app.mount("/static", StaticFiles(directory="static"), name="static")

# GLOBAL MODEL LOADING
model = None

@app.on_event("startup")
def load_model():
    global model
    try:
        # Make sure this matches your actual model filename!
        model = tf.keras.models.load_model('banana_model.keras')
        print("✅ TensorFlow Model Loaded Successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")

# 2. SERVE THE UI AT THE ROOT URL
@app.get("/")
async def read_root():
    # instead of returning JSON, we return the HTML file
    return FileResponse('static/index.html')

# HELPER FUNCTION
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# PREDICTION ENDPOINT
@app.post("/predict")
async def predict_banana(file: UploadFile = File(...)):
    if not model:
        return {"days_left": 0, "message": "Model not loaded. Check server logs."}

    try:
        image_bytes = await file.read()
        processed_image = preprocess_image(image_bytes)
        
        prediction = model.predict(processed_image)
        days_left = float(prediction[0][0])
        
        # Friendly rounding and logic
        days_left = round(max(0, days_left), 1)
        
        message = "Banana Bread Time! 🍞"
        if days_left > 7: message = "Green & Mean 🍏"
        elif days_left > 4: message = "Wait a bit... 🕒"
        elif days_left > 2: message = "Perfect! 🍌"
        elif days_left > 0.5: message = "Eat FAST! ⚠️"

        return {
            "days_left": days_left,
            "message": message
        }
    except Exception as e:
        return {"days_left": 0, "message": f"Error: {str(e)}"}