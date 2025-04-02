import librosa
import librosa.feature
import numpy as np
import pandas as pd
import os


#Mel-frequency cepstral coefficients (MFCCs)
# sr = numero de amostras por segundo
# librosa.feature.mfcc retorna uma matriz de dimensão (n_mfcc, T)
# n_mfcc=13 = O numero de coeficientes MFCC a serem extraídos
# T e o numero de quadros em que o audio foi dividido.
def extract_mfcc(y, sr, n_mfcc=13):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfccs, axis=1)  # Média dos coeficientes MFCC


# o centroide espectral indica onde a "energia" do espectro de frequência está concentrada.
# Diz se um som tende a ser mais grave ou mais agudo.
# Bateria ou baixos geralmente têm centroides espectrais baixos.
def extract_spectral_centroid(y, sr):
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    return np.mean(spectral_centroid)  # Média da centralidade espectral


# calcula a taxa de cruzamento por zero, ou seja, quantas vezes o sinal cruza o eixo zero ao longo do tempo.
# Sons com muitas variações rápidas (como instrumentos de percussão, ruídos ou sons agudos) tendem a ter ZCR alta.
# Sons suaves e contínuos (como vocais melódicos ou notas longas de instrumentos de corda) têm ZCR baixa.
def extract_zero_crossing_rate(y):
    zcr = librosa.feature.zero_crossing_rate(y)
    return np.mean(zcr)  # Média da taxa de cruzamento por zero

# Mede a intensidade percebida do som
def extract_loudness(y):
    rms = librosa.feature.rms(y=y)  # Calcula a energia RMS do áudio
    loudness = librosa.amplitude_to_db(rms)  # Converte para escala dB
    return np.mean(loudness)  # Retorna a média da sonoridade

# Mede o quão distribuída a energia está no espectro.
def extract_perceptual_spread(y, sr):
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spread = np.mean(spectral_bandwidth)  # Usa largura de banda espectral como estimativa
    return spread


# calcula a energia distribuída entre as 12 notas da escala musical (dó, dó#, ré, ré#...) ao longo do tempo.
#O resultado é um vetor de 12 valores, representando a intensidade média de cada nota na música.
# Ex: Músicas baseadas em acordes simples (ex: pop) terão picos em algumas notas específicas e estilos ricos em variação harmônica (ex: jazz, MPB) terão um perfil cromático mais distribuído.
def extract_chroma_features(y, sr):
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    return np.mean(chroma, axis=1)  # Média das características cromáticas


def extract_features(audio_path):
    y, sr = librosa.load(audio_path, mono=True)
    features = {}
    features['mfccs'] = extract_mfcc(y, sr).tolist()
    features['spectral_centroid'] = extract_spectral_centroid(y, sr)
    features['zero_crossing_rate'] = extract_zero_crossing_rate(y)
    features['loudness'] = extract_loudness(y)
    features['perceptual_spread'] = extract_perceptual_spread(y, sr)
    #features['chroma'] = extract_chroma_features(y, sr).tolist()
    return features

def process_dataset(audio_folder, output_csv):
    data = []
    for genre_folder in os.listdir(audio_folder):
        genre_path = os.path.join(audio_folder, genre_folder)
        if os.path.isdir(genre_path):
            for filename in os.listdir(genre_path):
                if filename.endswith(".wav"):
                    file_path = os.path.join(genre_path, filename)
                    features = extract_features(file_path)
                    row = {
                        'genre': genre_folder,
                        'filename': filename,
                        'spectral_centroid': features['spectral_centroid'],
                        'zero_crossing_rate': features['zero_crossing_rate'],
                        'loudness': features['loudness'],
                        'perceptual_spread': features['perceptual_spread']
                    }
                    row.update({f'mfcc_{i+1}': mfcc for i, mfcc in enumerate(features['mfccs'])})
                    #row.update({f'chroma_{i+1}': chroma for i, chroma in enumerate(features['chroma'])})
                    data.append(row)
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"Features salvas em {output_csv}")

# Exemplo de uso:
process_dataset("data/", "output4-forro30s.csv")
