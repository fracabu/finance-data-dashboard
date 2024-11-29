# 📊 Financial Analytics Dashboard - Premium Version 

## 🌟 [English]

### 🎯 Overview
A comprehensive financial analytics platform built with Streamlit, offering advanced data analysis capabilities, technical indicators, and portfolio management tools.

### ⚡ Quick Start
```bash
# Clone the repository
git clone [your-repo-url]

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run main.py
```

### 🚀 Features

#### 💼 Data Management
- 📁 Multi-format support (CSV, Excel, JSON)
- ⚡ Real-time data filtering
- 📅 Custom date range selection
- 🔄 Multi-symbol support

#### 📈 Analytics Features
- **📊 Main Dashboard**
  * Real-time visualization
  * Candlestick charts
  * Volume analysis
  * Moving averages

- **🔬 Advanced Analytics**
  * Correlation heatmaps
  * PCA Analysis
  * K-means clustering
  * Linear Regression
  * Statistical summaries

- **📉 Technical Analysis**
  * RSI
  * MACD
  * Bollinger Bands
  * Custom indicators

- **⚠️ Risk Analytics**
  * VaR calculations
  * Maximum Drawdown
  * Returns distribution
  * Risk metrics

### 🛠️ Technical Requirements
```txt
# Core packages
streamlit>=1.31.0
pandas>=2.2.0
numpy>=1.26.0

# Data visualization
matplotlib>=3.8.0
seaborn>=0.13.0
plotly>=5.18.0
```
[View full requirements.txt](requirements.txt)

---

## 🌟 [Italiano]

### 🎯 Panoramica
Una piattaforma completa di analisi finanziaria costruita con Streamlit, che offre capacità avanzate di analisi dati, indicatori tecnici e strumenti di gestione del portafoglio.

### ⚡ Avvio Rapido
```bash
# Clona il repository
git clone [url-repository]

# Installa le dipendenze
pip install -r requirements.txt

# Avvia l'applicazione
streamlit run main.py
```

### 🚀 Funzionalità

#### 💼 Gestione Dati
- 📁 Supporto multi-formato (CSV, Excel, JSON)
- ⚡ Filtraggio dati in tempo reale
- 📅 Selezione intervallo date personalizzato
- 🔄 Supporto multi-simbolo

#### 📈 Funzionalità Analitiche
- **📊 Dashboard Principale**
  * Visualizzazione in tempo reale
  * Grafici a candele
  * Analisi dei volumi
  * Medie mobili

- **🔬 Analisi Avanzata**
  * Mappe di correlazione
  * Analisi PCA
  * Clustering K-means
  * Regressione Lineare
  * Riassunti statistici

- **📉 Analisi Tecnica**
  * RSI
  * MACD
  * Bande di Bollinger
  * Indicatori personalizzabili

- **⚠️ Analisi del Rischio**
  * Calcoli VaR
  * Drawdown Massimo
  * Distribuzione rendimenti
  * Metriche di rischio

### 🛠️ Requisiti Tecnici
```txt
# Pacchetti principali
streamlit>=1.31.0
pandas>=2.2.0
numpy>=1.26.0

# Visualizzazione dati
matplotlib>=3.8.0
seaborn>=0.13.0
plotly>=5.18.0
```
[Visualizza requirements.txt completo](requirements.txt)

---

### 🌐 Links
- 📚 [Documentation](docs/)
- 🐛 [Issue Tracker](issues/)
- 📧 [Support](mailto:support@example.com)

### 📝 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---


# Comandi Base per Setup e Avvio Progetto

# 1. Creazione e attivazione ambiente virtuale
# Su Windows:
python -m venv venv
.\venv\Scripts\activate

# Su Mac/Linux:
python -m venv venv
source venv/bin/activate

# 2. Installazione dipendenze
pip install -r requirements.txt

# 3. Avvio applicazione
streamlit run app.py

# Comandi Aggiuntivi Utili

# Aggiornare pip
python -m pip install --upgrade pip

# Vedere lista pacchetti installati
pip list

# Creare/aggiornare requirements.txt
pip freeze > requirements.txt

# Disattivare ambiente virtuale
deactivate

# In caso di problemi con le dipendenze, reinstallazione pulita:
pip uninstall -r requirements.txt -y
pip install -r requirements.txt

# Per avviare su una porta specifica:
streamlit run app.py --server.port 8080

# Per abilitare il reload automatico:
streamlit run app.py --server.runOnSave true

# Per visualizzare informazioni di debug:
streamlit run app.py --logger.level=debug

# Per rendere l'app accessibile da altri dispositivi nella rete:
streamlit run app.py --server.address 0.0.0.0

# Per pulire la cache di Streamlit:
streamlit cache clear



# 1. Creazione della directory del progetto
C:\Users\utente>mkdir sales-dashboard
C:\Users\utente>cd sales-dashboard

# 2. Creazione dell'ambiente virtuale
C:\Users\utente\sales-dashboard>python -m venv venv

# 3. Attivazione dell'ambiente virtuale
C:\Users\utente\sales-dashboard>venv\Scripts\activate

# 4. Installazione delle dipendenze
(venv) C:\Users\utente\sales-dashboard>pip install -r requirements.txt

# 5. Avvio dell'applicazione
(venv) C:\Users\utente\sales-dashboard>streamlit run main.py
```

Questa è la sequenza esatta dall'inizio alla fine che ho seguito per:
1. Creare la cartella del progetto
2. Creare l'ambiente virtuale
3. Attivarlo
4. Installare le dipendenze 
5. Avviare l'app

Quando riavvii il PC o apri una nuova sessione del terminale, dovrai solo:
1. Navigare nella directory del progetto
2. Attivare l'ambiente virtuale 
3. Avviare l'app con l'ultimo comando

# 1. Accesso alla directory del progetto
C:\Users\utente>cd sales-dashboard

# 2. Attivazione ambiente virtuale
C:\Users\utente\sales-dashboard>venv\Scripts\activate

# 3. Avvio applicazione
(venv) C:\Users\utente\sales-dashboard>streamlit run main.py

Made with 🩸🩸 ⚡ 🔥 by fracabu 