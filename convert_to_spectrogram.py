import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Paths
input_path = r"C:\Users\reach\Desktop\scream_detector\data"  # Folder containing all class folders
output_path = r"C:\Users\reach\Desktop\scream_detector\data_spectrogram"  # Where spectrograms will be saved

# Ensure output directory exists
os.makedirs(output_path, exist_ok=True)

# Function to convert audio to spectrogram and save it
def save_spectrogram(audio_path, output_folder):
    y, sr = librosa.load(audio_path, sr=None)  # Load audio
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)  # Convert to decibels
    
    plt.figure(figsize=(5, 5))
    librosa.display.specshow(S_db, sr=sr, cmap='viridis')
    plt.axis("off")  # Remove axes
    
    # Save the spectrogram
    file_name = os.path.basename(audio_path).replace(".wav", ".png")
    os.makedirs(output_folder, exist_ok=True)  # Ensure class folder exists
    plt.savefig(os.path.join(output_folder, file_name), bbox_inches="tight", pad_inches=0)
    plt.close()

# Process all class folders
total_files = 0
for class_name in os.listdir(input_path):
    class_folder = os.path.join(input_path, class_name)
    if os.path.isdir(class_folder):  # Ensure it's a directory
        output_folder = os.path.join(output_path, class_name)
        os.makedirs(output_folder, exist_ok=True)
        
        for file in os.listdir(class_folder):
            if file.endswith(".wav"):
                save_spectrogram(os.path.join(class_folder, file), output_folder)
                total_files += 1

print(f"Spectrogram conversion completed! Processed {total_files} files.")
