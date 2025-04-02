import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

def load_dataset(csv_file):
    df = pd.read_csv(csv_file)
    X = df.drop(columns=['genre', 'filename'])  # Remove colunas não numéricas
    y = df['genre']
    return X, y


#Normaliza os valores das características usando StandardScaler.
def preprocess_data(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # Normaliza os dados
    
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)  # Converte os gêneros para valores numéricos
    
    return X_scaled, y_encoded, encoder

def train_knn_classifier(X_train, y_train, k=5):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    return knn

def evaluate_model(model, X_test, y_test, encoder):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))

def main(csv_file, k):
    X, y = load_dataset(csv_file)
    X_scaled, y_encoded, encoder = preprocess_data(X, y)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)
    
    knn_model = train_knn_classifier(X_train, y_train, k)
    evaluate_model(knn_model, X_test, y_test, encoder)
    
    return knn_model, encoder  # Retorna o modelo treinado e o encoder para uso futuro

# Exemplo de uso:
knn_model, encoder = main("output4-forro30s.csv", k=6)
