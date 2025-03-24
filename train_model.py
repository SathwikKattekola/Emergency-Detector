import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os

# Define paths
train_dir = r"C:\Users\reach\Desktop\scream_detector\split_data\train"
val_dir = r"C:\Users\reach\Desktop\scream_detector\split_data\val"

# Image properties
img_height, img_width = 128, 128
batch_size = 32

# Load dataset with multiple classes
train_ds = keras.preprocessing.image_dataset_from_directory(
    train_dir, image_size=(img_height, img_width), batch_size=batch_size, label_mode="categorical")

val_ds = keras.preprocessing.image_dataset_from_directory(
    val_dir, image_size=(img_height, img_width), batch_size=batch_size, label_mode="categorical")

# Get class names
class_names = train_ds.class_names
num_classes = len(class_names)
print(f"Classes: {class_names}")

# Normalize pixel values
normalization_layer = layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# Build CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),  # Prevents overfitting
    layers.Dense(num_classes, activation='softmax')  # Multi-class classification
])

# Compile model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',  # Multi-class classification
              metrics=['accuracy'])

# Early stopping to prevent overfitting
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=3, restore_best_weights=True)

# Train model
history = model.fit(train_ds, validation_data=val_ds, epochs=20, callbacks=[early_stopping])

# Save trained model
model_save_path = r"C:\Users\reach\Desktop\scream_detector\sound_classification_model.h5"
model.save(model_save_path)
print(f"Model saved successfully at {model_save_path}!")

# Plot training & validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training & Validation Loss')
plt.show()
