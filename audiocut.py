import librosa
import soundfile as sf
import os

# Trims the audio to a 15s sample taken from the middle of the file
def trim_audio(filepath, duration=15):
    y, sr = librosa.load(filepath, sr=16000)

    # Looks for the middle point and takes a 'duration' sample
    middle_point = len(y) // 2 
    end_point = middle_point + (sr * duration)

    # Makes sure the end_point < length
    y = y[middle_point:end_point] if end_point <= len(y) else y[middle_point:]

    # Saves new file
    sf.write(filepath, y, sr)
    print(f"Arquivo cortado (metade): {filepath}")


# path to genre folder
folder_path = "data/rock"  # Caminho da pasta com os arquivos

# Iterates over all the elements
for filename in os.listdir(folder_path):
    if filename.endswith(".wav"):  # Verifica se o arquivo é .wav
        file_path = os.path.join(folder_path, filename)
        trim_audio(file_path)  # Chama a função para cortar o áudio
