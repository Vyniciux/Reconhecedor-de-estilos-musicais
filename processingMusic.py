from scipy.signal import butter, filtfilt
import librosa
import numpy as np
import pandas as pd
import csv
import os

# Definir tamanho da janela e sobreposição
sample_rate = 16000  # Taxa de amostragem
frame_length = int(0.02 * sample_rate)  # 20ms em amostras
hop_length = int(frame_length * 0.4)  # 40% de sobreposição

def FourierTransform(music):
    # Aplicar a FFT normal

    fft = np.fft.fft(music)  # Computa a Transformada de Fourier
    magnitude_fft = np.abs(fft)  # Obtém a magnitude das frequências
    frequencies_fft = np.fft.fftfreq(len(magnitude_fft),1/sample_rate)  # Calcula os valores de frequência
    # len(magnitude) é o número de pontos da FFT (ou seja, o número de amostras do sinal de áudio)
    # 1/sr é o intervalo de tempo entre amostras (o "passo" no domínio do tempo)

    #Após calcular a FFT, essa função a simplifica calculando a média da magnitude em blocos de 200 amostras.
    
    # Tamanho do bloco
    tamanho_bloco = 10
    counter = 0
    freqs_ttf = []
    for i in range(200):
        media = 0
        for j in range(0,tamanho_bloco):
            media = media + magnitude_fft[counter+j]
        counter = counter + tamanho_bloco
        media = media/tamanho_bloco
        freqs_ttf.append(media)
        tamanho_bloco = tamanho_bloco + 4

    return freqs_ttf

def Filter(music, cutoff=40):
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

    n_mels=13 # Comprimento de salto
    mel_spectrogram = librosa.feature.melspectrogram(y=music, sr=sample_rate, n_fft=frame_length, hop_length=hop_length, n_mels=n_mels)

    # Converter para uma escala logarítmica (como o Log-Mel Spectrogram)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    return np.mean(mel_spectrogram, axis=1)  # Média ao longo do tempo

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

    return np.mean(chroma)

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

def mfcc(music, n_mfcc=13):
    #Mel-frequency cepstral coefficients (MFCCs)
    # sr = numero de amostras por segundo
    # librosa.feature.mfcc retorna uma matriz de dimensão (n_mfcc, T)
    # n_mfcc=13 = O numero de coeficientes MFCC a serem extraídos
    # T e o numero de quadros em que o audio foi dividido.
    mfccs = librosa.feature.mfcc(y=music, sr=sample_rate, n_mfcc=n_mfcc)
    return np.mean(mfccs, axis=1)  # Média dos coeficientes MFCC

def spectral_centroid(music):
    # o centroide espectral indica onde a "energia" do espectro de frequência está concentrada.
    # Diz se um som tende a ser mais grave ou mais agudo.
    # Bateria ou baixos geralmente têm centroides espectrais baixos.
    spectral_centroid = librosa.feature.spectral_centroid(y=music, sr=sample_rate)
    return np.mean(spectral_centroid)  # Média da centralidade espectral

def loudness(music):
    # Mede a intensidade percebida do som
    rms = librosa.feature.rms(y=music)  # Calcula a energia RMS do áudio
    loudness = librosa.amplitude_to_db(rms)  # Converte para escala dB
    return np.mean(loudness)  # Retorna a média da sonoridade

def perceptual_spread(music):
    # Mede o quão distribuída a energia está no espectro.
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=music, sr=sample_rate)
    spread = np.mean(spectral_bandwidth)  # Usa largura de banda espectral como estimativa
    return spread

def chroma_features(music):
    # calcula a energia distribuída entre as 12 notas da escala musical (dó, dó#, ré, ré#...) ao longo do tempo.
    #O resultado é um vetor de 12 valores, representando a intensidade média de cada nota na música.
    # Ex: Músicas baseadas em acordes simples (ex: pop) terão picos em algumas notas específicas e estilos ricos em variação harmônica (ex: jazz, MPB) terão um perfil cromático mais distribuído.
    chroma = librosa.feature.chroma_stft(y=music, sr=sample_rate)
    return chroma  # Média das características cromáticas

def spectral_rolloff(music):
    # Calcula o Spectral Roll-off
    rolloff = librosa.feature.spectral_rolloff(y=music, sr=sample_rate, roll_percent=0.75)
    return np.mean(rolloff)  # Retorna a média ao longo do tempo

def signal_energy(music):
    # Calcula a energia do sinal de áudio.
    # A energia é definida como a soma dos quadrados das amplitudes normalizadas pelo comprimento do sinal.
    energy = np.sum(music ** 2) / len(music)  # Energia média do sinal
    return energy

def ExtractToDatabase(audio_path, fileName, genre, db_name):
    # Carregar o arquivo de áudio
    #audio_path = "music.wav"  # Caminho do arquivo
    music, sr = librosa.load(audio_path, sr=sample_rate) # sr=None mantém a taxa de amostragem original, no caso fazendo um downsampling aqui.
    music = Filter(music)  # Filtragem para remover frequências baixas

    # Adicionar no dataset
    dado =( 
        genre,
        fileName,
        FourierTransform(music),
        ZeroCrossingRate(music),
        BeatPerSecond(music),
        spectral_centroid(music),
        loudness(music),
        perceptual_spread(music),
        mfcc(music),
    )

    with open(db_name +'.csv', mode='a', newline='', encoding='utf-8') as arquivo_csv:
        escritor_csv = csv.writer(arquivo_csv)
        escritor_csv.writerow(dado)
    print(f"=> {fileName} | {genre}")

def criar_dataset(db_name):
    # Criar DataFrame
    df = pd.DataFrame(dados)
    # Salvar em arquivo CSV
    df.to_csv(db_name+'.csv', index=False)

def process_all_music_files(data_folder, db_name):
    for folder in os.listdir(data_folder):
        folder_path = os.path.join(data_folder, folder)
        if os.path.isdir(folder_path):  # Verifica se é uma pasta
            for file in os.listdir(folder_path):
                if file.endswith(('.wav')):  # Adicione outros formatos de áudio, se necessário
                    print(f"Processando arquivo: {file}")
                    audio_path = os.path.join(folder_path, file)
                    ExtractToDatabase(audio_path, file, folder, db_name)

dados = {
    'genre': [],
    'song': [],
    'ttf': [],
    'zcr': [],
    'bps': [],
    'spectral_centroid': [],
    'loudness': [],
    'perceptual_spread': [],
    'mfccs': [],
    # Adicione outras características conforme necessário
}