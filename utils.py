import os
import yt_dlp
import soundfile as sf
import librosa
import ffmpeg
import glob

def download_music(music_url, output_dir):
    # Configurações do yt-dlp
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": f"{output_dir}/%(title)s.%(ext)s",
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",  # Baixa primeiro em MP3
            "preferredquality": "192",
        }],
    }

    # Baixa os arquivos de áudio
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([music_url])

def convert_mp3_to_wav(directory):
    # Converte MP3 para WAV usando soundfile
    count = 0 
    for file in os.listdir(directory):
        if file.endswith(".mp3"):
            mp3_path = os.path.join(directory, file)
            wav_path = os.path.join(directory, file.replace(".mp3", ".wav"))

            # Converte MP3 para WAV usando FFmpeg
            # Usando FFmpeg para converter MP3 para WAV (necessário para soundfile trabalhar com o WAV)
            audio_data, sample_rate = sf.read(mp3_path)  # Usando soundfile para ler
            sf.write(wav_path, audio_data, sample_rate)   # Usando soundfile para salvar em WAV

            # Opcional: remover o MP3 original após a conversão
            os.remove(mp3_path)
            print(f"{count} convertido: {file} para {wav_path}")

def trim_audio(filepath, duration=30):
    # Trims the audio to a 15s sample taken from the middle of the file
    y, sr = librosa.load(filepath, sr=16000)

    # Looks for the middle point and takes a 'duration' sample
    middle_point = len(y) // 2 
    end_point = middle_point + (sr * duration)

    # Makes sure the end_point < length
    y = y[middle_point:end_point] if end_point <= len(y) else y[middle_point:]

    # Saves new file
    sf.write(filepath, y, sr)
    print(f"Arquivo cortado (metade): {filepath}")

def trim_audio_3_in_parts(folder, duration=15):
    # Trims the audio to a 15s sample taken from the middle of the file

    filepath = ''
    for file in os.listdir(folder):
        if file.endswith(".wav"):
            filepath = os.path.join(folder, file)
            break  
    
    print(f"Arquivo: {filepath}")
    y, sr = librosa.load(filepath, sr=16000)

    length = len(y)/3
    # Looks for the middle point and takes a 'duration' sample
    for i in range(3):
        # Makes sure the end_point < length
        start = int(i*length)
        time = int(sr * duration)
        take = y[start:start+time] if start+start+time <= len(y) else y[start:]

        # Saves new file
        sf.write(f"{folder}/cut{i}.wav", take, sr)
    
    return filepath

def trim_all_folder(folder_path):
    # Iterates over all the elements
    for filename in os.listdir(folder_path):
        if filename.endswith(".wav"):  # Verifica se o arquivo é .wav
            file_path = os.path.join(folder_path, filename)
            trim_audio(file_path)  # Chama a função para cortar o áudio


def donwload_and_convert():
    # URL da playlist
    playlist_url = input("Qual o endereço da playlist? : ")
    # Diretório de saída
    output_dir = input("Qual o nome da nova pasta? : ")

    os.makedirs(output_dir, exist_ok=True)
    download_music(playlist_url, output_dir)
    print("Download concluído!")

    convert_mp3_to_wav(output_dir)
    print("Conversão concluída!")

def donwload_convert_and_cut():
    # URL da playlist
    playlist_url = input("Qual o endereço da playlist? : ")
    # Diretório de saída
    output_dir = input("Qual o nome da nova pasta? : ")

    os.makedirs(output_dir, exist_ok=True)
    download_music(playlist_url, output_dir)
    print("Download concluído!")

    convert_mp3_to_wav(output_dir)
    print("Conversão concluída!")

    trim_all_folder(output_dir)
    print("Corte concluído!")

