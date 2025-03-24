import os
import numpy as np
import librosa
import soundfile as sf

input_base_path = "janelamento_output/forro"
output_base_path = "hann_output"

# Output file structure
for root, dirs, files in os.walk(input_base_path):
    for dir_name in dirs:
        output_dir = root.replace(input_base_path, output_base_path)
        os.makedirs(os.path.join(output_dir, dir_name), exist_ok=True)

# Iterates over all files
for root, _, files in os.walk(input_base_path):
    for file in files:
        if file.endswith(".wav"):
            input_path = os.path.join(root, file)
            output_path = input_path.replace(input_base_path, output_base_path)

            # load the audio 20ms window
            y, sr = librosa.load(input_path, sr=None)

            # Applies Hann
            hann_window = np.hanning(len(y))
            y_hann = y * hann_window

            sf.write(output_path, y_hann, sr)
            print(f"Processado: {output_path}")
