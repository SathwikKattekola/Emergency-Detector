import sounddevice as sd
import numpy as np
import librosa
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("sound_classification_model.h5")
print("Model Loaded Successfully!")

# Add debugging information about model shapes
print(f"Model input shape: {model.input_shape}")
print(f"Model output shape: {model.output_shape}")

# Audio settings
SAMPLE_RATE = 22050  # Standard audio sample rate
DURATION = 3  # Record for 3 seconds

# Define class labels (adjust according to your model's training data)
CLASS_NAMES = ['car_crash', 'conversation', 'engine_idling', 'gun_shot', 'jambret', 'maling', 'rain', 'rampok', 'road_traffic', 'scream', 'thunderstorm', 'tolong', 'wind']
EMERGENCY_CLASSES = {"car_crash", "gun_shot", "scream"}  # Emergency cases

def preprocess_audio(audio, sample_rate):
    """Convert raw audio to Mel spectrogram for model input."""
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=128)
    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)  # Convert to decibels
    mel_spec = np.expand_dims(mel_spec, axis=-1)  # Add grayscale channel
    mel_spec = np.repeat(mel_spec, 3, axis=-1)  # Convert to 3-channel RGB
    mel_spec = np.expand_dims(mel_spec, axis=0)  # Add batch dimension
    
    # Add debugging information
    print(f"Processed input shape: {mel_spec.shape}")
    
    return mel_spec

def detect_emergency():
    """Continuously record audio and classify in real-time."""
    while True:
        print("\n🎤 Listening for sounds...")
        audio_data = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype=np.float32)
        sd.wait()  # Wait for recording to complete
        audio_data = np.squeeze(audio_data)  # Remove unnecessary dimensions

        # Preprocess audio
        input_data = preprocess_audio(audio_data, SAMPLE_RATE)

        # Predict using the model
        predictions = model.predict(input_data)[0]  # Get prediction array
        predicted_class_index = np.argmax(predictions)  # Get the highest probability class index
        predicted_class = CLASS_NAMES[predicted_class_index]  # Get class name

        # Print all class probabilities for debugging
        print("🔍 Class Probabilities:")
        for cls, prob in zip(CLASS_NAMES, predictions):
            print(f"  {cls}: {prob:.4f}")

        print(f"\n🎯 Prediction: {predicted_class} (Confidence: {predictions[predicted_class_index]:.4f})")

        # Check if it's an emergency class
        if predicted_class in EMERGENCY_CLASSES:
            print(f"🚨 Detected {predicted_class}! 🚨")
        else:
            print("✅ No emergency detected.")

# Run the detection loop
detect_emergency()