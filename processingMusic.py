from scipy.signal import butter, filtfilt
import librosa
import numpy as np
import pandas as pd
import csv
import os

# Definir tamanho da janela e sobreposição
sample_rate = 16000  # Taxa de amostragem
frame_length = int(0.02 * sample_rate)  # 20ms em amostras
hop_length = int(frame_length * 0.2)  # 20% de sobreposição

def FourierTransform(music):
    # Aplicar a FFT normal

    fft = np.fft.fft(music)  # Computa a Transformada de Fourier
    magnitude_fft = np.abs(fft)  # Obtém a magnitude das frequências
    frequencies_fft = np.fft.fftfreq(len(magnitude_fft),1/sample_rate)  # Calcula os valores de frequência
    # len(magnitude) é o número de pontos da FFT (ou seja, o número de amostras do sinal de áudio)
    # 1/sr é o intervalo de tempo entre amostras (o "passo" no domínio do tempo)

    #Após calcular a FFT, essa função a simplifica calculando a média da magnitude em blocos de 200 amostras.
    
    # Tamanho do bloco
    tamanho_bloco = 200

    freqs_ttf = []
    for i in range(200):
        media = 0
        for j in range(0,tamanho_bloco):
            media = media + magnitude_fft[tamanho_bloco*i+j]
        media = media/tamanho_bloco
        freqs_ttf.append(media)

    return freqs_ttf

def Filter(music, cutoff=50):
    # Filtragem (removendo frequências baixas que podem ser ruídos)
    # Remove frequências abaixo de cutoff Hz
    order=5
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    music = filtfilt(b, a, music)

    return music
    
def FastFourierTransform(music):

    # Aplicar STFT com janela Hamming
    stft = librosa.stft(music, n_fft=frame_length, hop_length=hop_length, window='hamming')

    #Pegar a média das magnitudes ao longo do tempo
    magnitude_stft = np.mean(np.abs(stft), axis=1)  # Média da magnitude para cada frequência

    # Converter para escala de magnitude
    stft_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)

    return stft_db

def MelSpectrogram(music):
    # Calcular o espectrograma de Mel

    # Definir tamanho da janela e sobreposição
    frame_length = int(0.02 * sample_rate)  # 20ms em amostras
    hop_length = int(frame_length * 0.2)  # 20% de sobreposição

    n_mels=100 # Comprimento de salto
    mel_spectrogram = librosa.feature.melspectrogram(y=music, sr=sample_rate, n_fft=frame_length, hop_length=hop_length, n_mels=n_mels)

    # Converter para uma escala logarítmica (como o Log-Mel Spectrogram)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    return log_mel_spectrogram

def ConstantQ(music):
    # Calcular o CQT
    # In mathematics and signal processing, the constant-Q transform and variable-Q transform, simply known as CQT and VQT, transforms a data series to the frequency domain. It is related to the Fourier transform[1] and very closely related to the complex Morlet wavelet transform.
    # https://en.wikipedia.org/wiki/Constant-Q_transform#:~:text=In%20mathematics%20and%20signal%20processing,the%20complex%20Morlet%20wavelet%20transform.

    cqt = librosa.cqt(music, sr=sample_rate, hop_length=hop_length)

    # Converter para uma escala logarítmica
    log_cqt = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)

    return log_cqt

def Chroma(music):
    # Chroma
    # The chroma feature is a descriptor, which represents the tonal content of a musical audio signal in a condensed form. Therefore chroma features can be considered as important prerequisite for high-level semantic analysis, like chord recognition or harmonic similarity estimation
    # https://en.wikipedia.org/wiki/Chroma_feature

    chroma = librosa.feature.chroma_stft(y=music, sr=sample_rate)

    return chroma

def ZeroCrossingRate(music):
    # Zero crossing rate
    # The zero crossing rate (ZCR) measures how many times the waveform crosses the zero axis.
    # https://www.sciencedirect.com/topics/engineering/zero-crossing-rate
    zcr = librosa.feature.zero_crossing_rate(music)
    zcr = np.mean(zcr)

    return zcr

def BeatPerSecond(music):
    # Calcular a envelope de onset
    # que representa a força dos ataques ao longo do tempo.
    onset_env = librosa.onset.onset_strength(y=music, sr=sample_rate)
    times = librosa.times_like(onset_env, sr=sample_rate, hop_length=hop_length)

    # Estimar o tempo e detectar batidas
    tempo, beat_dense = librosa.beat.beat_track(y=music, sr=sample_rate)
    # Converter os índices das batidas para tempos em segundos
    beat_times = librosa.frames_to_time(beat_dense, sr=sample_rate)


   # Quantas vezes o gráfico toca o eixo x por segundo

    bps = np.size(times[beat_dense])/(len(music) / sample_rate)

    return bps

def ExtractToDatabase(audio_path, fileName, gender):
    # Carregar o arquivo de áudio
    #audio_path = "music.wav"  # Caminho do arquivo
    music, sr = librosa.load(audio_path, sr=sample_rate) # sr=None mantém a taxa de amostragem original, no caso fazendo um downsampling aqui.
    music = Filter(music)  # Filtragem para remover frequências baixas

    # Adicionar no dataset
    dado =( 
        gender,
        fileName,
        FourierTransform(music),
        ZeroCrossingRate(music),
        BeatPerSecond(music),
    )

    with open("caracteristicas_musicas.csv", mode='a', newline='', encoding='utf-8') as arquivo_csv:
        escritor_csv = csv.writer(arquivo_csv)
        escritor_csv.writerow(dado)
    print(f"=> {fileName} | {gender}")

def criar_dataset():
    # Criar DataFrame
    df = pd.DataFrame(dados)
    # Salvar em arquivo CSV
    df.to_csv('caracteristicas_musicas.csv', index=False)

def process_all_music_files(data_folder):
    for folder in os.listdir(data_folder):
        folder_path = os.path.join(data_folder, folder)
        if os.path.isdir(folder_path):  # Verifica se é uma pasta
            for file in os.listdir(folder_path):
                if file.endswith(('.wav')):  # Adicione outros formatos de áudio, se necessário
                    print(f"Processando arquivo: {file}")
                    audio_path = os.path.join(folder_path, file)
                    ExtractToDatabase(audio_path, file, folder)

dados = {
    'gender': [],
    'song': [],
    'TTF': [],
    'ZCR': [],
    'BPS': [],
    # Adicione outras características conforme necessário
}

# Exemplo de uso
data_folder = "data"  # Caminho para a pasta contendo as músicas
criar_dataset()
process_all_music_files(data_folder)