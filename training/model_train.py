import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# 1. Setup Data Generators (with Augmentation)
# We use flow_from_dataframe since we have a CSV of regression labels
datagen = ImageDataGenerator(
    rescale=1./255,             # Normalize pixel values
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    validation_split=0.2        # Use 20% of data for validation
)

# Load CSV
df = pd.read_csv(r'C:\Users\HP\Desktop\ML\Machine Learning\Projects\AI_ML_Engineer\Days to the death of a banana(computer vision regression)\data\banana_days_471_synced.csv') # Columns: ['filename', 'days_left']
df['days_left'] = df['days_left'].astype(float) # Ensure regression targets are floats

# Training Data
train_generator = datagen.flow_from_dataframe(
    dataframe=df,
    directory=r'C:\Users\HP\Desktop\ML\Machine Learning\Projects\AI_ML_Engineer\Days to the death of a banana(computer vision regression)\data\banana_images_jpg',
    x_col='image_filename',
    y_col='days_left',
    target_size=(224, 224),
    batch_size=32,
    class_mode='raw', # 'raw' is essential for regression
    subset='training'
)

# Validation Data
val_generator = datagen.flow_from_dataframe(
    dataframe=df,
    directory=r'C:\Users\HP\Desktop\ML\Machine Learning\Projects\AI_ML_Engineer\Days to the death of a banana(computer vision regression)\data\banana_images_jpg',
    x_col='image_filename',
    y_col='days_left',
    target_size=(224, 224),
    batch_size=32,
    class_mode='raw',
    subset='validation'
)

# 2. Build the Model (Transfer Learning)
def build_model():
    # Load MobileNetV2 without the top classification layer
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Freeze the base model layers
    base_model.trainable = False

    # Add custom Regression Head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)  # Prevent overfitting
    x = Dense(128, activation='relu')(x)
    predictions = Dense(1, activation='linear')(x) # Linear activation for regression

    model = Model(inputs=base_model.input, outputs=predictions)
    return model

model = build_model()

# 3. Compile
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

# 4. Train
print("Starting training...")
history = model.fit(
    train_generator,
    epochs=15,
    validation_data=val_generator
)

# # 5. Fine-Tuning (Optional but recommended)
# # Unfreeze the last few layers of MobileNet for better accuracy
# base_model = model.layers[0]
# base_model.trainable = True
# # Fine-tune only the top 20 layers
# for layer in base_model.layers[:-20]:
#     layer.trainable = False

model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse', metrics=['mae'])
model.fit(train_generator, epochs=10, validation_data=val_generator)

# 6. Save for Production
# Saving in .keras format is the modern standard for TF 2.x
model.save('banana_model.keras')
print("Model saved as banana_model.keras")