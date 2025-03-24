import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Load the trained model
model = tf.keras.models.load_model("sound_classification_model.h5")
print("Model Loaded Successfully!")

# Define test data generator
test_datagen = ImageDataGenerator(rescale=1./255)

# Load test data
test_dir = "split_data/test"
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(128, 128),  # Match model input size
    batch_size=32,
    class_mode="categorical",  # Handles multiple classes
    shuffle=False
)

# Get class labels dynamically
class_labels = list(test_generator.class_indices.keys())

# Evaluate model
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

# Generate confusion matrix
y_true = test_generator.classes
y_pred = model.predict(test_generator)
y_pred = np.argmax(y_pred, axis=1)  # Convert probabilities to class labels
cm = confusion_matrix(y_true, y_pred)

# Save confusion matrix plot
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")  # Save as image
print("Confusion matrix saved as confusion_matrix.png")
