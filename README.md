
# CoinAnalys - Analyse en Temps Réel du Bitcoin 🪙

CoinAnalys est une application graphique qui fournit une analyse en temps réel du prix du Bitcoin, ainsi qu'un signal de prédiction basé sur un modèle d'apprentissage automatique. Ce projet est principalement une initiation à l'apprentissage machine (ML) et à l'apprentissage profond (DL), avec des prédictions basées sur des indicateurs techniques et un modèle LSTM.

---

## 🎨 Caractéristiques Principales

- **Interface graphique (GUI) avec Tkinter :**
  - Affichage dynamique des courbes de prix du Bitcoin.
  - Interface sombre pour une meilleure lisibilité.
  
- **Prédictions du modèle :**
  - Prédiction en temps réel basée sur un modèle d'apprentissage profond (LSTM).
  - Signal de prédiction ("Achat", "Vente", "Ne rien faire").
  
- **Données en temps réel :**
  - Les données sont récupérées via l'API CoinGecko, en particulier le prix du Bitcoin au cours des 365 derniers jours.
  
- **Mise à jour continue :**
  - Le graphique et les signaux sont mis à jour toutes les 60 secondes.

---

## 🛠️ Installation

### Prérequis
1. **Python 3.10+**
2. **Bibliothèques Python nécessaires :**
   - `numpy`
   - `pandas`
   - `matplotlib`
   - `tensorflow`
   - `sklearn`
   - `requests`
   - `tkinter`

Installez les dépendances avec la commande suivante :

```bash
pip install numpy pandas matplotlib tensorflow scikit-learn requests
```

---

## 🚀 Utilisation

1. Clonez le dépôt GitHub :
   ```bash
   git clone https://github.com/Eyub4k/CoinAnalys.git
   cd CoinAnalys
   ```

2. Lancez le fichier principal :
   ```bash
   python printing.py
   ```

3. **Navigation dans l'application :**
   - Vous verrez un graphique dynamique représentant l'évolution du prix du Bitcoin.
   - Un signal ("Achat", "Vente", "Ne rien faire") sera affiché en fonction de la prédiction du modèle.

---

## 🖥️ Fonctionnalités Techniques

### Récupération des données en temps réel

Les données sont récupérées via une API publique (CoinGecko) qui fournit les prix du Bitcoin pour les 365 derniers jours. Ces données sont ensuite transformées en un format structuré à l'aide de **pandas** pour un traitement ultérieur.

### Prétraitement des données et modèle de prédiction

Le modèle utilise des **indicateurs techniques** comme la moyenne mobile (MA7) et l'indicateur RSI (Relative Strength Index) pour prédire si le prix du Bitcoin va augmenter, baisser ou rester stable.

- **RSI** : un indicateur qui mesure la vitesse et le changement des mouvements de prix pour déterminer si un actif est suracheté ou survendu.
- **MA7** : la moyenne mobile sur 7 jours, utilisée pour lisser les fluctuations de prix à court terme.

Le modèle d'apprentissage profond est basé sur une architecture **LSTM** (Long Short-Term Memory), qui est bien adaptée aux séries temporelles.

Voici le code pour la prédiction et le prétraitement :

```python
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
```

### Modèle LSTM

Le modèle **LSTM** est conçu pour prédire la tendance future du prix du Bitcoin en se basant sur les données passées. Il utilise deux couches LSTM suivies de couches de régularisation (Dropout) pour éviter le surapprentissage.

```python
model = Sequential()

model.add(LSTM(units=200, activation='relu', input_shape=(window_size, len(features)), return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=100, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(units=3, activation='softmax'))  # Trois classes : Acheter, Vendre, Ne rien faire
```

### Précision du modèle

Il est important de noter que **la précision du modèle est d'environ 50%**, ce qui signifie que le modèle a une probabilité de prédiction de seulement 50% pour chaque catégorie. Cela signifie que le modèle est très basique et ne doit pas être utilisé pour prendre des décisions de trading en temps réel. Ce projet est un exemple d'initiation à l'apprentissage automatique et à l'apprentissage profond, et il est loin d'être un système de prédiction fiable à des fins réelles.


## 📂 Structure du Projet

```
CoinAnalys/
├── printing.py       # Script principal pour l'interface graphique et les mises à jour en temps réel
├── model.py          # Script contenant la définition et l'entraînement du modèle LSTM
├── crypto_model.h5   # Modèle pré-entraîné (LSTM)
└── README.md         # Documentation
```

---

## 💡 Améliorations Futures

- **Amélioration du modèle** : Augmenter la précision du modèle en optimisant les hyperparamètres et en utilisant davantage de données.
- **Ajout d'autres indicateurs techniques** : Intégration de nouveaux indicateurs tels que MACD, Bollinger Bands, etc.
- **Notifications** : Ajouter des alertes en temps réel pour les signaux "Achat" ou "Vente".

---

## 🧑‍💻 Contributeurs

- **Eyub4k** - Développeur Principal

---

## 📜 Licence

Ce projet est sous licence [MIT](LICENSE).
