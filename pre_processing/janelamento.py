import librosa
import numpy as np
import os
from scipy.io import wavfile

def janela_audio(arquivo_audio, pasta_saida="janelamento_output", duracao_janela_ms=20, sobreposicao=0.5):
    #caminho do arquivo
    caminho_relativo = os.path.relpath(arquivo_audio, "../data")
    genero, nome_musica = caminho_relativo.split(os.sep)[:2]
    nome_musica = os.path.splitext(nome_musica)[0]  # Remover extensão
    
    pasta_musica = os.path.join(pasta_saida, genero, nome_musica)
    os.makedirs(pasta_musica, exist_ok=True)
    
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
folder_path = "../data/forro"  # Substitua pelo caminho da sua pasta

# Iterates over all the elements
for filename in os.listdir(folder_path):
    if filename.endswith(".wav"):
        mp3_file_path = os.path.join(folder_path, filename)
        janelas, sr = janela_audio(mp3_file_path)