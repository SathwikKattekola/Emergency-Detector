import numpy as np
import librosa
import sounddevice as sd
import csv
import os

# Disable XNNPack delegate (optional for compatibility)
os.environ["TF_LITE_DISABLE_X86_NEON"] = "1"
os.environ["TF_DISABLE_XNNPACK"] = "1"

# Import TensorFlow Lite Interpreter
from tensorflow.lite.python.interpreter import Interpreter

# Load the TFLite model
interpreter = Interpreter(model_path='yamnet.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Determine model's expected input shape
expected_shape = input_details[0]['shape']
print(f"Model expects input shape: {expected_shape}")

# Load YAMNet class labels
def load_labels(csv_file):
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        return [row[2].strip().lower().replace(" ", "_") for row in reader]

class_names = load_labels("yamnet_class_map.csv")

# Target and interesting labels
TARGET = {"car_crash", "gun_shot", "scream"}
ALL = {"car_crash","conversation","engine_idling","gun_shot","rain","road_traffic","scream","thunderstorm","wind"}

# Record audio from microphone
def record_audio(duration=1, sr=44100):
    print("🎤 Recording...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    return np.squeeze(audio), sr

# Preprocess audio to match model requirements
def preprocess(audio, orig_sr, target_sr=16000, target_len=15600):
    if audio is None or len(audio) == 0:
        print("⚠️ Empty or invalid audio input")
        return None
    resampled = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr).astype(np.float32)
    if len(resampled) < target_len:
        resampled = np.pad(resampled, (0, target_len - len(resampled)))
    else:
        resampled = resampled[:target_len]
    return resampled

# Classify the processed waveform
def classify(waveform_16k):
    input_size = expected_shape[0]  # Usually 15600
    if len(waveform_16k) < input_size:
        waveform_16k = np.pad(waveform_16k, (0, input_size - len(waveform_16k)))
    else:
        waveform_16k = waveform_16k[:input_size]

    data = waveform_16k.reshape(expected_shape).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], data)
    interpreter.invoke()

    scores = interpreter.get_tensor(output_details[0]['index'])[0]
    top5 = np.argsort(scores)[::-1][:5]
    print("\n🧠 Top predictions:")
    for idx in top5:
        lbl = class_names[idx]
        conf = scores[idx]
        if lbl in ALL:
            prefix = "🚨 ALERT" if lbl in TARGET else "🔎"
            print(f"  {prefix}: {lbl} ({conf:.2f})")

# Main loop
if __name__ == "__main__":
    try:
        while True:
            audio, sr = record_audio()
            wave16 = preprocess(audio, sr)
            if wave16 is not None:
                classify(wave16)
            else:
                print("❌ Skipping classification due to invalid audio")
    except KeyboardInterrupt:
        print("\nExiting...")
