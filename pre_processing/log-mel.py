import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

input_base_path = "hann_output"
output_base_path = "log_mel_output"

# Parâmetros ajustados para janelas de 20ms
n_mels = 128  # Número de bandas Mel

def solve_path(input_path):
    caminho_relativo = os.path.relpath(input_path, input_base_path)
    genre, song, window = caminho_relativo.split(os.sep)
    window = os.path.splitext(window)[0]  # Remover extensão
        
    log_mel_directory = os.path.join(output_base_path, genre, song)
    os.makedirs(log_mel_directory, exist_ok=True)

    return genre, song, window


def log_mel(input_path):
    genre, song, window = solve_path(input_path)
    output_image_path = os.path.join(output_base_path, genre, song, window + ".png")
    output_npy_path = os.path.join(output_base_path, genre, song, window + ".npy")

    print(output_image_path)

    # Carregar o áudio
    y, sr = librosa.load(input_path, sr=None)

    # Definir tamanho da janela e hop_length baseado no janelamento de 20ms
    n_fft = int(sr * 0.020)  # 20ms janela
    hop_length = n_fft // 2  # Passo entre janelas = 50% da janela
    fmax = sr // 2  # Cobrir toda a faixa audível

    # Normalizar o áudio
    y = y / np.max(np.abs(y))

    # Calcular o Log-Mel Spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmax=fmax)
    mel_spec = np.where(mel_spec == 0, 1e-10, mel_spec)  # Evitar zeros no espectrograma
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)  # Converter para dB

    # Salvar como numpy array para uso futuro
    np.save(output_npy_path, log_mel_spec)

    # Visualizar e salvar como imagem
    plt.figure(figsize=(5, 4))
    librosa.display.specshow(log_mel_spec, sr=sr, hop_length=hop_length, fmax=fmax, cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Log-Mel Spectrogram: {1}')
    plt.savefig(output_image_path, bbox_inches='tight')
    plt.close()

    print(f"✅ Processado: {output_image_path}")


# path to genre folder
folder_path = "hann_output/forro/A Gente Se Entrega"

# Iterates over all the elements
for window in os.listdir(folder_path):
    if window.endswith(".wav"):
        wav_file_path = os.path.join(folder_path, window).replace("\\", "/")
        log_mel(wav_file_path)