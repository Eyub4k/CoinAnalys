import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import time
import requests
import pandas as pd
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Charger le modèle
model = load_model('crypto_model.h5')

# Fonction pour obtenir des données en temps réel
def get_historical_data():
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {
        "vs_currency": "usd",
        "days": "365",
        "interval": "daily"
    }
    response = requests.get(url, params=params)
    data = response.json()

    if 'prices' not in data:
        print("Erreur : 'prices' non trouvé dans la réponse de l'API.")
        return None

    prices = data['prices']
    df = pd.DataFrame(prices, columns=["timestamp", "price"])
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.drop('timestamp', axis=1, inplace=True)
    return df

# Prétraitement et prédiction
def preprocess_and_predict(df):
    window_size = 30
    df['MA7'] = df['price'].rolling(window=7).mean()
    df['RSI'] = (df['price'] - df['price'].shift(1)).rolling(window=14).apply(
        lambda x: (x[x > 0].sum() / abs(x).sum()) * 100 if abs(x).sum() != 0 else 50
    )
    df = df.dropna()
    last_data = df[['price', 'MA7', 'RSI']].tail(window_size)
    scaler = MinMaxScaler(feature_range=(0, 1))
    last_scaled = scaler.fit_transform(last_data)
    X_realtime = last_scaled.reshape((1, window_size, len(last_data.columns)))
    prediction = model.predict(X_realtime)
    predicted_class = np.argmax(prediction)
    return "Achat" if predicted_class == 1 else "Vente" if predicted_class == 0 else "Ne rien faire"

# Mise à jour de l'interface
def update_interface():
    global df
    df = get_historical_data()
    if df is None or df.empty:
        return

    signal = preprocess_and_predict(df)
    last_price = df['price'].iloc[-1]
    price_label.config(text=f"Prix actuel : ${last_price:,.2f}")
    prediction_label.config(text=f"Signal : {signal}", fg="green" if signal == "Achat" else "red" if signal == "Vente" else "white")

    # Nettoyage du graphique avant de redessiner
    ax.clear()
    
    # Tracer les données
    ax.plot(df['date'].values, df['price'].values, color="cyan", label="Prix BTC")
    ax.scatter(df['date'].iloc[-1], df['price'].iloc[-1], color="yellow", label="Prix actuel", zorder=5)
    
    # Configurer l'arrière-plan et le style des axes
    ax.set_title("Prix Bitcoin (USD)", color="white")
    ax.set_facecolor("#1c1c1c")  # Fond des axes
    ax.tick_params(colors="white")
    ax.legend(loc="upper left", facecolor="#1c1c1c", edgecolor="white")
    
    # Configurer l'arrière-plan de la figure
    fig.patch.set_facecolor("#1c1c1c")  # Fond du canvas (tout le graphique)
    fig.patch.set_alpha(1.0)  # Assurez-vous qu'il est complètement opaque

    # Dessiner le graphique sur le canvas
    canvas.draw()

    # Mise à jour automatique toutes les 60 secondes
    root.after(60000, update_interface)


# Initialisation de l'interface graphique
root = tk.Tk()
root.title("Analyse du Bitcoin")
root.configure(bg="#1c1c1c")

# Titre
title_label = tk.Label(root, text="Analyse en Temps Réel du Bitcoin", font=("Helvetica", 20), bg="#1c1c1c", fg="white")
title_label.pack(pady=10)

# Label du prix actuel
price_label = tk.Label(root, text="Prix actuel : $0.00", font=("Helvetica", 16), bg="#1c1c1c", fg="cyan")
price_label.pack(pady=5)

# Label de prédiction
prediction_label = tk.Label(root, text="Signal : En attente...", font=("Helvetica", 16), bg="#1c1c1c", fg="white")
prediction_label.pack(pady=5)

# Graphique
fig = Figure(figsize=(10, 6), dpi=100)
ax = fig.add_subplot(111)
ax.set_facecolor("#1c1c1c")
ax.tick_params(colors="white")

canvas = FigureCanvasTkAgg(fig, master=root)
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack()

# Lancement de la mise à jour
update_interface()

# Lancer l'interface Tkinter
root.mainloop()
