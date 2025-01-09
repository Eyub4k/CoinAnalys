import requests
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.metrics import classification_report

# 1. Télécharger les données historiques de crypto (Bitcoin ici)
def get_historical_data():
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {
        "vs_currency": "usd",
        "days": "365",
        "interval": "daily"
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    prices = data['prices']
    
    df = pd.DataFrame(prices, columns=["timestamp", "price"])
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.drop('timestamp', axis=1, inplace=True)
    
    return df

df = get_historical_data()

# 2. Ajouter des indicateurs techniques (RSI et MA7)
df['MA7'] = df['price'].rolling(window=7).mean()

def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi

df['RSI'] = calculate_rsi(df['price'])

# 3. Calculer la colonne Target (Acheter = 1, Ne rien faire = 2, Vendre = 0)
def create_target_column(df):
    target = []
    for i in range(1, len(df)):
        if df['price'][i] > df['price'][i - 1]:
            target.append(1)  # Acheter
        elif df['price'][i] < df['price'][i - 1]:
            target.append(0)  # Vendre
        else:
            target.append(2)  # Ne rien faire
    return target

# Ajouter un 2 initial pour la première ligne
df['Target'] = [2] + create_target_column(df)

# Vérifier si la colonne 'Target' existe maintenant
print(df.head())

# 4. Normalisation des prix
scaler = MinMaxScaler(feature_range=(0, 1))
df['scaled_price'] = scaler.fit_transform(df['price'].values.reshape(-1, 1))

# Préparer les données pour LSTM
window_size = 30
features = ['scaled_price', 'MA7', 'RSI']
X = df[features].values
y = df['Target'].values

# 5. Créer les séquences temporelles avec TimeseriesGenerator
generator = TimeseriesGenerator(X, y, length=window_size, batch_size=32)

model = Sequential()

# Première couche LSTM avec un plus grand nombre d'unités et retour des séquences
model.add(LSTM(units=200, activation='relu', input_shape=(window_size, len(features)), return_sequences=True))
model.add(Dropout(0.2))  # Dropout pour éviter le surapprentissage

# Deuxième couche LSTM avec plus d'unités
model.add(LSTM(units=100, activation='relu'))
model.add(Dropout(0.2))  # Dropout supplémentaire

# Couche Dense pour la classification
model.add(Dense(units=3, activation='softmax'))  # Trois classes

# Compiler avec un taux d'apprentissage plus bas
model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Entraîner avec un plus grand nombre d'époques et un meilleur batch_size
history = model.fit(generator, epochs=100, batch_size=64, verbose=1)

# Prédictions et évaluation
predictions = model.predict(generator)
predictions_classes = np.argmax(predictions, axis=1)

# Évaluer avec classification report
print(classification_report(y[window_size:], predictions_classes))

# Afficher la précision globale
from sklearn.metrics import accuracy_score
print(f"Accuracy globale: {accuracy_score(y[window_size:], predictions_classes)}")

