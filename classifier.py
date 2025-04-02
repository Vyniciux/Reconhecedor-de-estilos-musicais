import pandas as pd
import numpy as np
import ast
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC  # Importando SVC
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import processingMusic

# Carregar os dados do CSV
try:
    df = pd.read_csv("caracteristicas_musicas.csv")
except FileNotFoundError:
    print("Arquivo 'caracteristicas_musicas.csv' não encontrado. Criando o dataset...")
    processingMusic.criar_dataset("caracteristicas_musicas")
    processingMusic.process_all_music_files("data", "caracteristicas_musicas")
    df = pd.read_csv("caracteristicas_musicas.csv")

def corrigir_formato_mfccs(texto):
    # Remove quebras de linha e múltiplos espaços corretamente
    texto_corrigido = re.sub(r'\s+', ' ', texto.strip())  # Remove espaços extras
    texto_corrigido = texto_corrigido.replace(" ", ",")   # Converte espaços em vírgulas
    
    # Corrige o caso de uma vírgula extra no início
    if texto_corrigido.startswith("[,"):
        texto_corrigido = "[" + texto_corrigido[2:]

    try:
        return np.array(ast.literal_eval(texto_corrigido), dtype=np.float64)
    except (SyntaxError, ValueError) as e:
        print(f"Erro ao processar: {texto} -> {e}")
        return np.nan  # Retorna NaN para valores problemáticos


def test_acuracy():

    # Preparar X (características) e y (rótulos)
    X_base = df.drop(columns=['genre', 'song', 'ttf', 'mfccs'])  # Remove colunas não numéricas

    # Converte 'ttf' de string para array NumPy de forma segura
    X_ttf = df['ttf'].apply(lambda x: np.array(eval(x.replace("np.float64(", "").replace(")", ""))))

    X_mfccs = df['mfccs'].apply(corrigir_formato_mfccs)

    # Concatena as features base com as extraídas de 'ttf'
    X = np.hstack((X_base.values, np.vstack(X_ttf.values), np.vstack(X_mfccs.values)))

    y = df['genre']

    # Normalizar as características
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Codificar os rótulos
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    # Aumentar os dados das classes menos representadas (se possível)
    smote = SMOTE(random_state=90)
    X_scaled, y_encoded = smote.fit_resample(X_scaled, y_encoded)

    # Dividir em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=80)

    # Criar e treinar o modelo SVC
    svc = SVC(kernel='rbf', C=40, gamma='scale')  # SVC com kernel RBF
    svc.fit(X_train, y_train)

    # Fazer previsões
    y_pred = svc.predict(X_test)

    # Converter os rótulos de volta para os nomes dos gêneros
    y_test_labels = encoder.inverse_transform(y_test)
    y_pred_labels = encoder.inverse_transform(y_pred)

    # Avaliar o modelo
    accuracy = accuracy_score(y_test, y_pred)

    # Printar a tabela corretamente
    print("Acurácia do modelo:", accuracy)
    print(classification_report(y_test_labels, y_pred_labels))



def test_genre(data_path, song_name):

    # Preparar X (características) e y (rótulos)
    X_base = df.drop(columns=['genre', 'song', 'ttf', 'mfccs'])  # Remove colunas não numéricas

    # Converte 'ttf' de string para array NumPy de forma segura
    X_ttf = df['ttf'].apply(lambda x: np.array(eval(x.replace("np.float64(", "").replace(")", ""))))

    X_mfccs = df['mfccs'].apply(corrigir_formato_mfccs)
    # Concatena as features base com as extraídas de 'ttf'
    X = np.hstack((X_base.values, np.vstack(X_ttf.values), np.vstack(X_mfccs.values)))

    y = df['genre']

    # Normalizar as características
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Codificar os rótulos
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    # Aumentar os dados das classes menos representadas (se possível)
    smote = SMOTE(random_state=90)
    X_scaled, y_encoded = smote.fit_resample(X_scaled, y_encoded)

    # Criar e treinar o modelo SVC
    svc = SVC(kernel='rbf', C=40, gamma='scale')
    svc.fit(X_scaled, y_encoded)

    # --- Predição para uma nova música ---
    # Carregar os dados da nova música
    data = pd.read_csv(data_path)

    data_base = data.drop(columns=['genre', 'song', 'ttf', 'mfccs'])  # Remove colunas não numéricas

    # Converte 'ttf' de string para array NumPy de forma segura
    data_ttf = data['ttf'].apply(lambda x: np.array(eval(x.replace("np.float64(", "").replace(")", ""))))

    data_mfccs = data['mfccs'].apply(corrigir_formato_mfccs)

    # Certifique-se de que a forma do array é consistente
    d = np.hstack((data_base.values, np.stack(data_ttf.values), np.stack(data_mfccs.values)))

    # Ajusta a forma do array caso tenha múltiplas amostras
    if d.ndim == 1:
        d = d.reshape(1, -1)

    # Normalizar as características da nova música
    d_scaled = scaler.transform(d)

    # Obter distâncias aos hiperplanos das classes para as 3 amostras
    distances = svc.decision_function(d_scaled)

    distances_mean = np.sum(distances, axis=0).reshape(1, -1)        
    print(distances_mean)

    # Obter os 5 gêneros mais prováveis
    top_indices = np.argsort(distances_mean[0])[-5:][::-1]

    top_genres = encoder.inverse_transform(top_indices)
    # Exibir 
    print('####################################################################')
    print(f"#####################{song_name}########################")
    print()
    print(f"1- Gênero previsto: {top_genres[0]}!")
    print()
    print(f"2- Tem uma chance alta de ser {top_genres[1]}.")
    print(f"3- Talvez seja {top_genres[2]} também, vai.")
    print(f"4- Poder ser {top_genres[3]}...")
    print(f"5- Um chance pequena para {top_genres[4]}...")
    print()
    print('####################################################################')


