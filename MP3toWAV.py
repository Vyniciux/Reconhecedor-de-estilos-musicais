from pydub import AudioSegment
import os

# Converts a MP3 file into WAV
def convert_mp3_to_wav(mp3_path):
    wav_path = mp3_path.replace(".mp3", ".wav")
    
    sound = AudioSegment.from_mp3(mp3_path)
    sound.export(wav_path, format="wav")
    print(f"Convertido: {mp3_path} → {wav_path}")

    # Removes the mp3 file after conversion
    os.remove(mp3_path)
    print(f"Arquivo MP3 removido: {mp3_path}")

# path to genre folder
folder_path = "data/forro"  # Substitua pelo caminho da sua pasta

# Iterates over all the elements
for filename in os.listdir(folder_path):
    if filename.endswith(".mp3"):
        mp3_file_path = os.path.join(folder_path, filename)
        convert_mp3_to_wav(mp3_file_path)
