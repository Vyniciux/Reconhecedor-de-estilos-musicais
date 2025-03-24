import librosa
import numpy as np
import os
from scipy.io import wavfile

input_base_path = "../data"
output_base_path = "janelamento_output"

def solve_path(input_path):
    caminho_relativo = os.path.relpath(input_path, input_base_path)
    genre, song = caminho_relativo.split(os.sep)
    song = os.path.splitext(song)[0]  # Remover extensão
        
    log_mel_directory = os.path.join(output_base_path, genre, song)
    os.makedirs(log_mel_directory, exist_ok=True)
    output_dir = os.path.join(output_base_path, genre, song)
    return output_dir


def janela_audio(arquivo_audio, duracao_janela_ms=20, sobreposicao=0.5):
    pasta_musica = solve_path(arquivo_audio)
    
    y, sr = librosa.load(arquivo_audio, sr=None)
    
    # Definir o tamanho da janela em amostras
    tamanho_janela = int((duracao_janela_ms / 1000) * sr)
    tamanho_passo = int(tamanho_janela * (1 - sobreposicao))
    
    # Criar as janelas usando uma matriz
    janelas = librosa.util.frame(y, frame_length=tamanho_janela, hop_length=tamanho_passo).T
    
    # Salvar cada janela como um novo arquivo WAV
    for i, janela in enumerate(janelas):
        arquivo_saida = os.path.join(pasta_musica, f"janela_{i}.wav")
        wavfile.write(arquivo_saida, sr, (janela * 32767).astype(np.int16))  # Converter para int16
    
    print(f"{len(janelas)} janelas salvas em {pasta_musica}")
    return janelas, sr

# path to genre folder
folder_path = "../data/rock"  # Substitua pelo caminho da sua pasta

# Iterates over all the elements in the directory
# for filename in os.listdir(folder_path):
#     if filename.endswith(".wav"):
#         wav_file_path = os.path.join(folder_path, filename).replace("\\", "/")
#         #print(wav_file_path)
#         janelas, sr = solve_path(wav_file_path)


# Only one song for test purposes
janelas, sr = janela_audio("../data/rock/Cabrobró.wav")
