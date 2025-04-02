import librosa
import soundfile as sf
import os

# Trims the audio to a 15s sample taken from the middle of the file
def trim_audio(filepath, duration=30):
    y, sr = librosa.load(filepath, sr=16000)

    # Looks for the middle point and takes a 'duration' sample
    middle_point = len(y) // 2 
    end_point = middle_point + (sr * duration)

    # Makes sure the end_point < length
    y = y[middle_point:end_point] if end_point <= len(y) else y[middle_point:]

    # Saves new file
    sf.write(filepath, y, sr)
    print(f"Arquivo cortado (metade): {filepath}")


# Trims the audio to a 30s sample centered on the middle of the file
def trim_audio_15sModule(filepath, duration=30):
    y, sr = librosa.load(filepath, sr=16000)

    # Calculate the middle point
    middle_point = len(y) // 2

    # Calculate the start and end points for the 30-second window
    half_duration_samples = (sr * duration) // 2
    start_point = middle_point - half_duration_samples
    end_point = middle_point + half_duration_samples

    # Ensure the start and end points are within the audio bounds
    start_point = max(0, start_point)
    end_point = min(len(y), end_point)

    # Trim the audio
    y = y[start_point:end_point]

    # Save the new file
    sf.write(filepath, y, sr)
    print(f"Arquivo cortado (centrado na metade): {filepath}")

# path to genre folder
folder_path = "data/forro"  # Caminho da pasta com os arquivos

# Iterates over all the elements
for filename in os.listdir(folder_path):
    if filename.endswith(".wav"):  # Verifica se o arquivo é .wav
        file_path = os.path.join(folder_path, filename)
        trim_audio(file_path)  # Chama a função para cortar o áudio
