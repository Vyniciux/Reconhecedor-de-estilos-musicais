import os
import numpy as np
import librosa
import soundfile as sf

input_base_path = "janelamento_output"
output_base_path = "hann_output"

def solve_path(input_path):
    caminho_relativo = os.path.relpath(input_path, input_base_path)
    genre, song, window = caminho_relativo.split(os.sep)
    window = os.path.splitext(window)[0]  # Remover extensão
        
    log_mel_directory = os.path.join(output_base_path, genre, song)
    os.makedirs(log_mel_directory, exist_ok=True)

    output_dir = os.path.join(output_base_path, genre, song, window)
    return output_dir


def hann(input_path):
    # Iterates over all files
    output_path = solve_path(input_path) + ".wav"

    # load the audio 20ms window
    y, sr = librosa.load(input_path, sr=None)

    # Applies Hann
    hann_window = np.hanning(len(y))
    y_hann = y * hann_window

    sf.write(output_path, y_hann, sr)
    print(f"Processado: {output_path}")


# path to genre folder
folder_path = "janelamento_output/rock"

#Iterates over all the windows in a single song for testing purposes
# for window in os.listdir(folder_path):
#     if window.endswith(".wav"):
#         wav_file_path = os.path.join(folder_path, window).replace("\\", "/")
#         hann(wav_file_path)

# Iterantes over all the songs in a genre dir
for song in os.listdir(folder_path):
    song_folder_path = os.path.join(folder_path, song)
    for window in os.listdir(song_folder_path):
        wav_file_path = os.path.join(folder_path, song, window).replace("\\", "/")
        hann(wav_file_path)
