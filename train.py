import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
import numpy as np
import os
import json

# -----------------------------python
# Parameters
# -----------------------------
dataset_path = "dataset"  # folder seen in your VS Code screenshot
img_size = (224, 224)
batch_size = 16
epochs = 10
tflite_model_path = "plant_model.tflite"
class_names_path = "classes.json"
disease_info_path = "disease_info.json"

# -----------------------------
# Dataset Preparation
# -----------------------------
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# -----------------------------
# Class Names
# -----------------------------
class_names = list(train_generator.class_indices.keys())
print("Classes found:", class_names)

# Save class names to JSON
with open(class_names_path, "w") as f:
    json.dump(class_names, f, indent=4)
print(f"Class names saved to {class_names_path}")

# -----------------------------
# Disease Info Mapping
# -----------------------------
disease_info = {
    "Blight": {
        "Name": "Northern Corn Leaf Blight",
        "Causes": "Fungal disease caused by Exserohilum turcicum",
        "Prevention": "Use resistant hybrids, rotate crops, and remove crop debris",
        "Treatment": "Apply fungicides such as strobilurins or triazoles",
        "Plant Health": "Diseased"
    },
    "Common_Rust": {
        "Name": "Common Rust",
        "Causes": "Fungal disease caused by Puccinia sorghi",
        "Prevention": "Plant resistant varieties and avoid dense planting",
        "Treatment": "Use fungicides (triazoles, strobilurins) during early infection",
        "Plant Health": "Diseased"
    },
    "Gray_Leaf_Spot": {
        "Name": "Cercospora Leaf Spot (Gray Leaf Spot)",
        "Causes": "Fungal disease caused by Cercospora zeae-maydis",
        "Prevention": "Crop rotation and resistant varieties",
        "Treatment": "Apply triazole-based fungicides and improve air circulation",
        "Plant Health": "Diseased"
    },
    "Healthy": {
        "Name": "Healthy Corn Leaf",
        "Causes": "None — plant is healthy",
        "Prevention": "Maintain proper irrigation and nutrient management",
        "Treatment": "No treatment required",
        "Plant Health": "Healthy"
    },
    "Tomato_Bacterial_spot": {
        "Name": "Bacterial Spot",
        "Causes": "Bacterial infection caused by Xanthomonas campestris pv. vesicatoria",
        "Prevention": "Use certified seeds, avoid overhead watering, rotate crops",
        "Treatment": "Copper-based bactericides and removal of infected leaves",
        "Plant Health": "Diseased"
    },
    "Tomato_Early_blight": {
        "Name": "Early Blight",
        "Causes": "Fungal disease caused by Alternaria solani",
        "Prevention": "Crop rotation, avoid wet leaves, and use resistant varieties",
        "Treatment": "Apply fungicides (chlorothalonil, copper oxychloride)",
        "Plant Health": "Diseased"
    },
    "Tomato_Late_blight": {
        "Name": "Late Blight",
        "Causes": "Fungal-like pathogen Phytophthora infestans",
        "Prevention": "Use certified disease-free seeds and avoid waterlogging",
        "Treatment": "Apply fungicides (mancozeb, metalaxyl) and destroy infected plants",
        "Plant Health": "Diseased"
    },
    "Tomato_Leaf_Mold": {
        "Name": "Leaf Mold",
        "Causes": "Fungal disease caused by Passalora fulva",
        "Prevention": "Increase air circulation and avoid high humidity in greenhouses",
        "Treatment": "Use sulfur or copper fungicides and prune infected leaves",
        "Plant Health": "Diseased"
    },
    "Tomato_Septoria_leaf_spot": {
        "Name": "Septoria Leaf Spot",
        "Causes": "Fungal disease caused by Septoria lycopersici",
        "Prevention": "Avoid overhead irrigation and practice crop rotation",
        "Treatment": "Apply fungicides (chlorothalonil, mancozeb)",
        "Plant Health": "Diseased"
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "Name": "Two-Spotted Spider Mite Infestation",
        "Causes": "Infestation by Tetranychus urticae mites",
        "Prevention": "Maintain proper humidity and avoid drought stress",
        "Treatment": "Use miticides or neem oil and remove heavily infested leaves",
        "Plant Health": "Diseased"
    },
    "Tomato_healthy": {
        "Name": "Healthy Tomato Leaf",
        "Causes": "None — plant is healthy",
        "Prevention": "Ensure proper sunlight, watering, and nutrition",
        "Treatment": "No treatment required",
        "Plant Health": "Healthy"
    }
}

# Save disease info to JSON
with open(disease_info_path, "w") as f:
    json.dump(disease_info, f, indent=4)
print(f"Disease info saved to {disease_info_path}")

# -----------------------------
# Model Building
# -----------------------------
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
predictions = Dense(len(class_names), activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.summary()

# -----------------------------
# Training
# -----------------------------
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs
)

# -----------------------------
# Save Model
# -----------------------------
model.save("plant_model.h5")
print("Keras model saved as plant_model.h5")

# -----------------------------
# Convert to TFLite
# -----------------------------
print("Converting model to TFLite format...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)

print(f"TFLite model saved as {tflite_model_path}")
